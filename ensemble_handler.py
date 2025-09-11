import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import helpers


def loss_function(weights, probs_list, labels):
    weighted_probs = np.average(probs_list, axis=0, weights=weights)
    sum_of_probs = np.sum(weighted_probs, axis=1, keepdims=True)
    normalized_probs = weighted_probs / sum_of_probs
    return log_loss(labels, normalized_probs)


def ensemble_handler(model_probs_list, test_loader):
    """
    Given a list of model, the function optimize their weights for a final prediction
    :param model_probs_list:
    :param test_loader:
    """
    helpers.print_section("ENSEMBLE WEIGHTED")

    labels, _ = get_all_labels(data_loader=test_loader)
    num_models = len(model_probs_list)
    opt_weights = optimize.minimize(
        loss_function,
        np.array([1.0 / num_models] * num_models),
        args=(model_probs_list, labels),
        constraints=({'type': 'eq', 'fun': lambda w: 1 - sum(w)}),
        method='SLSQP',
        bounds=[(0.05, 1.0)] * num_models,
        options={'ftol': 1e-10},
    )['x']

    combined_probs = np.average(model_probs_list, axis=0, weights=opt_weights)
    sum_of_probs = np.sum(combined_probs, axis=1, keepdims=True)
    normalized_probs = combined_probs / sum_of_probs

    final_bin_prev = np.argmax(normalized_probs, axis=1)
    ensemble_accuracy = accuracy_score(labels, final_bin_prev)
    ensemble_log_loss = log_loss(labels, normalized_probs)

    with open(r'ensemble_weights.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(opt_weights, file)

    print(
        f"Optimal weight: {opt_weights}\nEnsemble accuracy: {ensemble_accuracy:.4f}%\nEnsemble Log Loss: {ensemble_log_loss:.4f}")


def ensemble_results(model_probs_list, data_loader):
    """
    Given the models, generate a final weighted prediction for the set
    :param model_probs_list:
    :param data_loader:
    """
    helpers.print_section("WEIGHTED ENSEMBLER")

    # Load weights
    try:
        with open('ensemble_weights.pkl', 'rb') as file:
            optimal_weights = pickle.load(file)
    except FileNotFoundError:
        print(f"Error while loading ensemble weight")
        return None

    labels, _ = get_all_labels(data_loader=data_loader)

    combined_probs = np.average(model_probs_list, axis=0, weights=optimal_weights)
    sum_of_probs = np.sum(combined_probs, axis=1, keepdims=True)
    normalized_probs = combined_probs / sum_of_probs

    final_bin_prev = np.argmax(normalized_probs, axis=1)
    ensemble_accuracy = accuracy_score(labels, final_bin_prev)
    ensemble_log_loss = log_loss(labels, normalized_probs)

    print(f"Ensemble accuracy: {ensemble_accuracy * 100:.2f}%\nEnsemble log loss: {ensemble_log_loss:.4f}")


def get_all_labels(data_loader):
    """
    Extract all the label from the DataLoader (Args)
    :return: np.Array containing all the label of DataLoader.
    """
    all_labels = []
    all_labels_tensor = []
    for batch in data_loader:
        labels = batch['label']
        all_labels.extend(labels.cpu().numpy())
        all_labels_tensor.append(labels.cpu())
    return np.array(all_labels), torch.cat(all_labels_tensor, dim=0)


def get_model_probs(model, dataloader, device):
    """
    Generate the prediction probability of a give model on the dataloader
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def ensemble_majority_voting(model_probe_list, data_loader):
    helpers.print_section("MAJORITY VOTING ENSEMBLER")

    labels, _ = get_all_labels(data_loader=data_loader)
    integer_model_probe_list = [np.round(model_list).astype(int) for model_list in model_probe_list]

    probe_tensor = torch.tensor(np.array(integer_model_probe_list), dtype=torch.int64)

    final_predictions = []
    for i in range(probe_tensor.shape[1]):
        example_probs = probe_tensor[:, i]

        sum_of_votes = torch.sum(example_probs).item()

        if sum_of_votes >= 2:
            final_predictions.append(1)
        else:
            final_predictions.append(0)

    correct_predictions = sum(1 for pre, label in zip(final_predictions, labels) if pre == label)
    ensemble_accuracy = correct_predictions / len(labels) if len(labels) else 0
    print(f"Ensemble accuracy: {ensemble_accuracy * 100:.2f}%")


def ensemble_meta_model(train_features, train_loader, test_features, test_loader, num_epochs, batch_size):
    helpers.print_section("ENSEMBLE META MODEL")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    train_features = torch.cat(train_features, dim=1)
    test_features = torch.cat(test_features, dim=1)
    _, train_labels = get_all_labels(train_loader)
    _, test_labels = get_all_labels(test_loader)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    meta_model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        meta_model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = meta_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    meta_model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = meta_model(features)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Meta model accuracy: {accuracy * 100:.2f}")

    torch.save(meta_model.state_dict(), "meta_model_weights.pth")


def ensemble_meta_results(test_features, test_loader, batch_size):
    helpers.print_section("ENSEMBLE META RESULTS")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    test_features = torch.cat(test_features, dim=1)
    _, test_labels = get_all_labels(test_loader)
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = test_features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    weights_path = Path("meta_model_weights.pth")
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print("Meta model loaded with saved weights")
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Meta model accuracy: {accuracy * 100:.2f}%")
