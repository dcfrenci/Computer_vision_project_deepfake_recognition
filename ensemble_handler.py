from scipy import optimize
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import pickle
import torch
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
    labels = get_all_labels(data_loader=test_loader)
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

    labels = get_all_labels(data_loader=data_loader)

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
    for batch in data_loader:
        labels = batch['label']
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels)


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

    labels = get_all_labels(data_loader=data_loader)
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
    # ensemble_log_loss = log_loss(labels, correct_predictions)

    # print(f"Ensemble accuracy: {ensemble_accuracy:.4f}%\nEnsemble log loss: {ensemble_log_loss:.4f}")
    print(f"Ensemble accuracy: {ensemble_accuracy * 100:.2f}%")
