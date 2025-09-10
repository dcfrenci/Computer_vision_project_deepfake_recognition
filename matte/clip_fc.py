import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import helpers
from pathlib import Path


class DeepfakeClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.head = nn.Linear(512, 2)

    def forward(self, image):
        features = self.clip.encode_image(image)
        logits = self.head(features.float())

        return logits


def clip_handler(train_loader, test_loader,num_epochs):
    helpers.print_section("CLIP")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for param in clip_model.parameters():
        param.requires_grad = False

    model = DeepfakeClassifier(clip_model).to(device)

    if Path("matte/fc_layer_weight.pth").exists():
        model.head.load_state_dict(torch.load("matte/fc_layer_weight.pth", weights_only=True, map_location=device))
        print(f"head layer loaded with pretrained weights")
    else:
        print(f"head layer loaded without pretrained weights")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)

    # --- Main Training Loop ---
    outputs = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, outputs = evaluate_model(model, test_loader, criterion, device)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Save just the weight and bias of all layer
    torch.save(model.head.state_dict(), "matte/fc_layer_weight.pth")

    return outputs


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


def clip_fc_results(data_loader):
    helpers.print_section("CLIP")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for param in clip_model.parameters():
        param.requires_grad = False

    model = DeepfakeClassifier(clip_model).to(device)

    if Path("matte/fc_layer_weight.pth").exists():
        model.head.load_state_dict(torch.load("matte/fc_layer_weight.pth", weights_only=True, map_location=device))
        print(f"head layer loaded with pretrained weights")
    else:
        print(f"head layer loaded without pretrained weights")

    criterion = nn.CrossEntropyLoss()

    # --- Main Training Loop ---
    epoch_loss, accuracy, outputs = evaluate_model(model, data_loader, criterion, device)
    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return outputs


def clip_fc_get_features(data_loader):

    helpers.print_section("CLIP GET FEATURES")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for param in clip_model.parameters():
        param.requires_grad = False

    all_features = []
    with torch.no_grad():
        for input, labels in data_loader:
            input = input.to(device)
            features = clip_model.encode_image(input)
            all_features.append(features)
    
    return all_features