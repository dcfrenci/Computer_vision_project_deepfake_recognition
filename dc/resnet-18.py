import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import requests
from io import BytesIO
from torch.utils.data import Subset

# Import images
elsa_data = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
elsa_data_test = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)

# Resize images for the resnet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# def collate_fn(batch):
#     images = []
#     labels = []
#     for example in batch:
#         # for key in ["image_gen0", "image_gen1", "image_gen2", "image_gen3"]:
#         for key in ["image_gen0"]:
#             if example[key] is not None:
#                 images.append(transform(example[key]))
#                 labels.append(0)
#                 # labels.append(example["label"])  # stessa label per tutte
#
#     # Produces batch_size * 4 images (The same image from different generator)
#     return torch.stack(images), torch.tensor(labels)


def collate_fake_real(batch):
    images = []
    labels_ = []

    for example in batch:  # example è già un singolo dict
        if random.random() < 0.5:  # testa/croce
            # Croce: prendiamo image_gen0 (fake)
            img = example["image_gen0"]  # già un'immagine PIL
            if img is not None:
                images.append(transform(img))
                labels_.append(0)
        else:
            # Testa: scarichiamo immagine reale da url
            url = example["url"]
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(transform(img))
                labels_.append(1)
                print("OK download")
            except Exception as e:
                # fallback: se non riesce a scaricare, prendiamo image_gen0
                print(f"NOT SO OK download: {e}")
                img = example["image_gen0"]  # già PIL Image
                if img is not None:
                    images.append(transform(img))
                    labels_.append(0)

    # torch.stack richiede tensori tutti della stessa dimensione
    return torch.stack(images), torch.tensor(labels_)


# Create a DataLoader that applies the transform on-the-fly
train_loader = DataLoader(Subset(elsa_data, range(64)), batch_size=32, collate_fn=collate_fake_real)
print("Train_loader complete")
test_loader = DataLoader(Subset(elsa_data_test, range(64)), batch_size=32, collate_fn=collate_fake_real)
print("Test_loader complete")

print(train_loader)

# Split the dataset into train, validation, test
model = resnet18(weights="IMAGENET1K_V1")  # Pretrained
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # adjust epochs
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
