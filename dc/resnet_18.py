import helpers
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path


# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy, all_outputs


def resnet_handler(train_loader, test_loader, num_epochs):
    helpers.print_section("RESNET 18")

    # --- Model, Loss, and Optimizer Definition ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = resnet18() if Path("dc/resnet_18_weight.pth").exists() else resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 2)
    model = model.to(device)
    if Path("dc/resnet_18_weight.pth").exists():
        model.load_state_dict(torch.load("dc/resnet_18_weight.pth", weights_only=True, map_location=device))
        print(f"Resnet loaded with pretrained weights")
    else:
        print(f"Resnet loaded with IMAGENET1K_V1 weights")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Main Training Loop ---
    outputs = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, outputs = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Save just the weight and bias of all layer
    torch.save(model.state_dict(), "dc/resnet_18_weight.pth")

    return outputs


def resnet_results(data_loader):
    helpers.print_section("RESNET 18")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = resnet18()
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 2)
    weights_path = Path("dc/resnet_18_weight.pth")
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print("Resnet loaded with saved weights")
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None

    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    epoch_loss, accuracy, outputs = evaluate_model(model, data_loader, criterion, device)
    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return outputs


# def resnet_feature_training(train_loader, test_loader, num_epochs):
#     helpers.print_section("RESNET FEATURE TRAINING")
#
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     print(f"Using device: {device}")
#
#     model = resnet18() if Path("dc/resnet_feature_extractor_weights.pth").exists() else resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#
#     num_feature = model.fc.in_features
#     model.fc = nn.Linear(num_feature, 2)
#     model = model.to(device)
#     if Path("dc/resnet_feature_extractor_weights.pth").exists():
#         model.load_state_dict(torch.load("dc/resnet_feature_extractor_weights.pth", weights_only=True, map_location=device))
#         print(f"Resnet loaded with pretrained weights")
#     else:
#         print(f"Resnet loaded with IMAGENET1K_V1 weights")
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # --- Main Training Loop ---
#     for epoch in range(num_epochs):
#         train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
#         test_loss, test_accuracy, _ = evaluate_model(model, test_loader, criterion, device)
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
#
#     # Save just the weight and bias of all layer removed the last one
#     torch.save(model.state_dict(), "dc/resnet_feature_extractor_weights.pth")


def resnet_get_features(data_loader):
    helpers.print_section("RESNET GET FEATURE")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = resnet18()
    model = nn.Sequential(*(list(model.children())[:-1]))
    weights_path = Path("dc/resnet_18_weight.pth")
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print("Resnet loaded with saved weights")
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None

    model = model.to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)
