import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import helpers

def ciao_metodo():
    print("ciao")

def clip_handler(train_loader, test_loader):
    helpers.print_section("CLIP")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for param in clip_model.parameters():
        param.requires_grad = False
    
    class DeepfakeClassifier(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip = clip_model
            self.head = nn.Linear(512, 2)
        
        def forward(self, image):

            features = self.clip.encode_image(image)
            logits = self.head(features.float())

            return logits

    model = DeepfakeClassifier(clip_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)

    # --- Main Training Loop ---
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Save just the weight and bias of all layer
    torch.save(model.state_dict(), "matte/fc_layer_weight.pth")



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

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy