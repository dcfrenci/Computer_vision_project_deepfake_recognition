import helpers
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
import numpy as np
from torchvision import transforms
import pywt
import cv2


# =========================
# Wavelet preprocessing
# =========================
def apply_wavelet(batch_images):
    batch_wavelet = []

    for img in batch_images:  # img: [3,H,W]
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        comps = []

        for c in range(3):  # R,G,B
            cA, (cH, cV, cD) = pywt.dwt2(img_np[:, :, c], "haar")

            for comp in [cA, cH, cV, cD]:
                comp_resized = cv2.resize(comp, (224, 224))
                comps.append(comp_resized)

        # shape: [12,H,W]
        wavelet_img = np.stack(comps, axis=0)
        wavelet_tensor = torch.from_numpy(wavelet_img).float()

        # Normalizzazione tipo ImageNet (ripetuta per ogni gruppo di 3 canali)
        mean = [0.485, 0.456, 0.406] * 4
        std = [0.229, 0.224, 0.225] * 4
        wavelet_tensor = transforms.Normalize(mean=mean, std=std)(wavelet_tensor)

        batch_wavelet.append(wavelet_tensor)

    return torch.stack(batch_wavelet)  # [B,12,224,224]


# =========================
# ResNet18 modificata a 12 canali
# =========================
def build_resnet18_12ch(pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # salva conv1 originale
    old_conv = model.conv1

    # nuovo conv1 con 12 canali
    model.conv1 = nn.Conv2d(12, old_conv.out_channels,
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride,
                            padding=old_conv.padding,
                            bias=old_conv.bias)

    if pretrained:
        # inizializza i pesi copiando i 3 canali originali su tutti i 12
        with torch.no_grad():
            model.conv1.weight[:, :3, :, :] = old_conv.weight
            for i in range(1, 4):
                model.conv1.weight[:, 3*i:3*(i+1), :, :] = old_conv.weight

    return model


# =========================
# Training per un'epoca
# =========================
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


# =========================
# Valutazione del modello
# =========================
def evaluate_model_wavelet(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

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

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


# =========================
# Pipeline completa
# =========================
def frequency_handler(train_loader, test_loader):
    helpers.print_section("FREQUENCY DECOMPOSITION + RESNET 18 (12ch)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # costruisci il modello 12ch
    model = build_resnet18_12ch(pretrained=True)
    num_feature = model.fc.in_features
    model.fc = nn.Linear(num_feature, 2)
    model = model.to(device)

    # carica pesi se già salvati
    weight_path = Path("simo/frequency_resnet_18_weight.pth")
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("ResNet18 caricata con pesi salvati")
    else:
        print("ResNet18 caricata con pesi ImageNet adattati a 12 canali")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch_wavelet(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate_model_wavelet(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Acc: {test_accuracy:.2f}%")

    # salva i pesi
    torch.save(model.state_dict(), weight_path)
    return 0
