from pathlib import Path

import clip
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_grad_cam import GradCAM
from sklearn.manifold import TSNE
from torchvision import transforms

import helpers


class DeepfakeClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.head = nn.Linear(512, 2)

    def forward(self, image):
        features = self.clip.encode_image(image)
        logits = self.head(features.float())

        return logits


def clip_handler(train_loader, test_loader, num_epochs, path_name):
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

    if Path(path_name).exists():
        model.head.load_state_dict(torch.load(path_name, weights_only=True, map_location=device))
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
    torch.save(model.head.state_dict(), path_name)

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


def clip_fc_results(data_loader, path_name):
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

    if Path(path_name).exists():
        model.head.load_state_dict(torch.load(path_name, weights_only=True, map_location=device))
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
        for batch in data_loader:
            inputs = batch['image'].to(device)
            inputs = inputs.to(device)
            features = clip_model.encode_image(inputs)
            all_features.append(features)

    return torch.cat(all_features, dim=0)


def clip_heatmap_handler(path_name):
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

    if Path(path_name).exists():
        model.head.load_state_dict(torch.load(path_name, weights_only=True, map_location=device))
        print(f"head layer loaded with pretrained weights")
    else:
        print(f"head layer loaded without pretrained weights")
    model.to(device).eval()
    model.float()
    image_path = "xception_model/images/fotopersona.jpeg"
    original_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor = input_tensor.float().requires_grad_(True)
    target_layers = [model.clip.visual.conv1]
    # Inizializza Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    helpers.heatmap_helpers(cam, model, input_tensor, original_image)


def clip_fc_plot_tsne(data_loader, path_name=None):
    helpers.print_section("CLIP t-SNE")

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

    if path_name and Path(path_name).exists():
        model.head.load_state_dict(torch.load(path_name, map_location=device))
        print("head layer loaded with pretrained weights")
    else:
        print("head layer loaded without pretrained weights")

    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image'].to(device)
            labels = batch['label']
            features = clip_model.encode_image(inputs)
            logits = model.head(features.float())

            all_features.append(logits.cpu())
            all_labels.append(labels.cpu())

    x = torch.cat(all_features, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    x_2d = tsne.fit_transform(x)

    # plotting
    plt.figure(figsize=(7, 6))
    plt.scatter(x_2d[y == 0, 0], x_2d[y == 0, 1], c='blue', s=5, label="Real", alpha=0.6)
    plt.scatter(x_2d[y == 1, 0], x_2d[y == 1, 1], c='red', s=5, label="Fake", alpha=0.6)
    plt.legend()
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    title = "CLIP (Pretrained)" if path_name and Path(path_name).exists() else "CLIP (Untrained)"
    plt.title(title)
    plt.show()
