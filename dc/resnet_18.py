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


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy, all_outputs


def resnet_handler(train_loader, test_loader):
    helpers.print_section("RESNET 18")

    # --- Model, Loss, and Optimizer Definition ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    num_epochs = 10
    prob_outputs = None
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, prob_outputs = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Save just the weight and bias of all layer
    torch.save(model.state_dict(), "dc/resnet_18_weight.pth")

    return prob_outputs
