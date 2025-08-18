from PIL import Image
from io import BytesIO
import requests
import random
import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights


class CustomListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_example(example_pr):
    img = None
    label = None

    if random.random() < 0.5:
        try:
            url = example_pr["url"]
            response = requests.get(url, timeout=2)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            label = 1
        except Exception:
            img = example_pr["image_gen0"]
            if isinstance(img, bytes):
                img = Image.open(BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")
            label = 0
    else:
        img = example_pr["image_gen0"]
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        label = 0

    if img:
        img = transform(img)

    return {'image': img, 'label': label}


# Load datasets in streaming mode
elsa_data = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
elsa_data_test = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)

# Sample a subset of examples
subset_train = []
for i, example in enumerate(elsa_data):
    if i >= 64:
        break
    subset_train.append(process_example(example))

subset_test = []
for i, example in enumerate(elsa_data_test):
    if i >= 64:
        break
    subset_test.append(process_example(example))

print(f"Sampled {len(subset_train)} examples for training.")
print(f"Sampled {len(subset_test)} examples for testing.")

# Wrap the lists in the custom Dataset class
train_dataset = CustomListDataset(subset_train)
test_dataset = CustomListDataset(subset_test)

# Create the DataLoader using the Dataset instances
train_loader = DataLoader(train_dataset, batch_size=32)
print("Train_loader ready")

test_loader = DataLoader(test_dataset, batch_size=32)
print("Test_loader ready")

# --- Model, Loss, and Optimizer Definition ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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


# --- Main Training Loop ---
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
