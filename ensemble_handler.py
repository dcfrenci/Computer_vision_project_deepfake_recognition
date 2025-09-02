import numpy as np
from scipy import optimize
from sklearn.metrics import log_loss, accuracy_score
import pickle
import torch


def loss_function(weights, probs_list, labels):
    weighted_probs = np.average(probs_list, axis=0, weights=weights)
    return log_loss(labels, weighted_probs)


def ensemble_handler(model_probs_list, test_loader):
    """
    Given a list of model it optimize their weights for a final prediction
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
        bounds=[(0.0, 1.0)] * num_models,
        options={'ftol': 1e-10},
    )['x']

    combined_probs = np.average(model_probs_list, axis=0, weights=opt_weights)
    final_bin_prev = np.argmax(combined_probs, axis=1)
    ensemble_accuracy = accuracy_score(labels, final_bin_prev)
    ensemble_log_loss = log_loss(labels, combined_probs)

    with open(r'ensemble_weights.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(opt_weights, file)

    print(
        f"Optimal weight: {opt_weights}\nEnsemble accuracy: {ensemble_accuracy:.4f}\nEnsemble Log Loss: {ensemble_log_loss:.4f}")


def ensemble_evaluation(models, data_loader):
    """
    Given the models, generate a final weighted prediction for the set
    :param models:
    :param data_loader:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load weights
    try:
        with open('ensemble_weights.pkl', 'rb') as file:
            optimal_weights = pickle.load(file)
    except FileNotFoundError:
        print(f"Error while loading ensemble weight")
        return None

    model_probs_list = []
    for model in models:
        probs = get_model_probs(model, data_loader, device)
        model_probs_list.append(probs)

    labels = get_all_labels(data_loader=data_loader)

    final_preds_proba = np.average(model_probs_list, axis=0, weights=optimal_weights)
    final_preds = np.argmax(final_preds_proba, axis=1)
    ensemble_accuracy = accuracy_score(labels, final_preds)
    ensemble_log_loss = log_loss(labels, final_preds_proba)

    print(f"Ensemble accuracy: {ensemble_accuracy}\nEnsemble log loss: {ensemble_log_loss}")


def get_all_labels(data_loader):
    """
    Extract all the label from the DataLoader (Args)
    :return:np.array: An array containing all the label of DataLoader.
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
