from pathlib import Path

import cv2
import numpy as np
import pywt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import helpers



# Wavelet preprocessing
def apply_wavelet(batch_images):
    batch_wavelet = []

    for img in batch_images:
        # [C,H,W]-[H,W,C]
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        comps = []

        for c in range(3):
            cA, (cH, cV, cD) = pywt.dwt2(img_np[:, :, c], "haar")

            for comp in [cA, cH, cV, cD]:
                comp_resized = cv2.resize(comp, (
                224, 224))
                comps.append(comp_resized)
        # shape: [12,H,W]
        wavelet_img = np.stack(comps, axis=0)
        wavelet_tensor = torch.from_numpy(wavelet_img).float()

        mean = [0.485, 0.456, 0.406] * 4
        std = [0.229, 0.224, 0.225] * 4
        wavelet_tensor = transforms.Normalize(mean=mean, std=std)(wavelet_tensor)

        batch_wavelet.append(wavelet_tensor)
    return torch.stack(batch_wavelet)



# Xception modified to 12 channels
def build_xception_12ch(weight_path: Path):
    if weight_path.exists():
        model = timm.create_model("legacy_xception", pretrained=False, num_classes = 2)
        old_conv = model.conv1  #reference to the original layer
        model.conv1 = nn.Conv2d(12, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias)
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("Xception loaded with pretrained weights")
    else:
        # ImageNet-pretrained weights
        model = timm.create_model("legacy_xception", pretrained=True, num_classes=2)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(12, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias)
        with torch.no_grad():
            # Replicate 3-channel weights across 12 input channels
            model.conv1.weight[:, :3, :, :] = old_conv.weight
            for i in range(1, 4):
                model.conv1.weight[:, 3 * i:3 * (i + 1), :, :] = old_conv.weight
        print("Xception loaded with ImageNet weights adapted to 12 channels")
    return model


def train_epoch_wavelet(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        inputs = apply_wavelet(inputs).to(device)  # [B,12,224,224]

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_model_wavelet(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    all_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            inputs = apply_wavelet(inputs).to(device)  # [B,12,224,224]
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


def frequency_handler(train_loader, test_loader, num_epochs, path_name):
    helpers.print_section("FREQUENCY DECOMPOSITION + Xception (12ch)")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    weight_path = Path(path_name)

    model = build_xception_12ch(weight_path)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training
    outputs = []

    for epoch in range(num_epochs):
        train_loss = train_epoch_wavelet(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, outputs = evaluate_model_wavelet(model, test_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Acc: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), weight_path)
    return outputs


def frequency_results(data_loader, path_name):
    helpers.print_section("FREQUENCY + Xception 18")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    weight_path = Path(path_name)
    if weight_path.exists():
        model = build_xception_12ch(weight_path)
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    epoch_loss, accuracy, outputs = evaluate_model_wavelet(model, data_loader, criterion, device)
    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return outputs


#--------------------------------LAST_LAYER_REMOVED-----------------------------------------------
def xception_feature_extractor(data_loader, path_name):
    helpers.print_section("Xception model feature extractor")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    weight_path = Path(path_name)
    if weight_path.exists():
        model = build_xception_12ch(weight_path)
        model = nn.Sequential(*(list(model.children())[:-1]))
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None
    model.to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = apply_wavelet(batch['image']).to(device)
            features = model(inputs)
            features = features.view(features.size(0), -1)
            all_features.append(features)
    return torch.cat(all_features, dim=0)

#------------------------------------------------HEATMAP---------------------------------------------------
def xception_heatmap_handler(path_name):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    weight_path = Path(path_name)
    if not weight_path.exists():
        print("Error: Saved weights not found. Please train the model first.")
        return

    model = build_xception_12ch(weight_path)
    model.to(device).eval()

    image_path = "xception_model/images/fotopersona.jpeg"
    original_image = Image.open(image_path).convert("RGB")

    transform_for_wavelet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor_for_wavelet = transform_for_wavelet(original_image).unsqueeze(0).to(device)
    input_tensor_12ch = apply_wavelet(input_tensor_for_wavelet).to(device)
    target_layers = [model.conv4]
    #print(model)
    cam = GradCAM(model=model, target_layers=target_layers)
    helpers.heatmap_helpers(cam, model,input_tensor_12ch,original_image)


def frequency_plot_tsne(data_loader, path_name=None):
    helpers.print_section("XCEPTION TSNE")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    weight_path = Path(path_name)
    if weight_path.exists():
        model = build_xception_12ch(weight_path)
        model = nn.Sequential(*(list(model.children())[:-1]))
    else:
        print("Error: Saved weights not found. Please train the model first.")
        return None
    model.to(device)
    model.eval()

    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = apply_wavelet(batch['image']).to(device)
            labels = batch['label']
            features = model(inputs)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())


    X = torch.cat(all_features, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)


    # plotting
    plt.figure(figsize=(7,6))
    plt.scatter(X_2d[y==0,0], X_2d[y==0,1], c='blue', s=5, label="Real", alpha=0.6)
    plt.scatter(X_2d[y==1,0], X_2d[y==1,1], c='red', s=5, label="Fake", alpha=0.6)
    plt.legend()
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    title = "XCEPTION (Pretrained)" if path_name and Path(path_name).exists() else "XCEPTION (Untrained)"
    plt.title(title)
    plt.show()