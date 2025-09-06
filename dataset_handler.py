import random
import helpers
import requests
from tqdm import tqdm
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


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

CATCHABLE_EXCEPTIONS = (
    requests.exceptions.RequestException,
    IOError,
    UnidentifiedImageError
)


def process_example(example_pr):
    img = None
    label = None
    if random.random() < 0.5:
        try:
            url = example_pr["url"]
            response = requests.get(url, timeout=2)
            img = Image.open(BytesIO(response.content)).convert("RGBA").convert("RGB")
            label = 1
        except CATCHABLE_EXCEPTIONS:
            img = example_pr["image_gen0"]
            if isinstance(img, bytes):
                img = Image.open(BytesIO(img)).convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGBA").convert("RGB")
            label = 0
    else:
        img = example_pr["image_gen0"]
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img)).convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGBA").convert("RGB")
            label = 0

    if img:
        img = transform(img)
    return {'image': img, 'label': label}


def dataset_handler(num_train_examples, num_test_examples, batch_size):
    helpers.print_section("DATASET")

    print(f"Sampling {num_train_examples} for training and {num_test_examples} for testing")

    # Load datasets in streaming mode
    elsa_data = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
    elsa_data_test = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)

    # Sample a subset of examples
    subset_train = []
    for i, example in enumerate(tqdm(elsa_data, total=num_train_examples, desc="Training Data")):
        if i >= num_train_examples:
            break
        subset_train.append(process_example(example))

    subset_test = []
    for i, example in enumerate(tqdm(elsa_data_test, total=num_test_examples, desc="Test Data    ")):
        if i >= num_test_examples:
            break
        subset_test.append(process_example(example))

    print(f"Sampling completed")

    # Wrap the lists in the custom Dataset class
    train_dataset = CustomListDataset(subset_train)
    test_dataset = CustomListDataset(subset_test)

    # Create the DataLoader using the Dataset instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(f"Train and test loader ready (batch size = {batch_size})")
    return train_loader, test_loader


def dataset_results(num_test_examples, batch_size):
    helpers.print_section("DATASET")
    print(f"Sampling {num_test_examples} for testing")

    elsa_data_test = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)

    subset_test = []
    for i, example in enumerate(tqdm(elsa_data_test, total=num_test_examples, desc="Test Data    ")):
        if i >= num_test_examples:
            break
        subset_test.append(process_example(example))

    print(f"Sampling completed")
    test_dataset = CustomListDataset(subset_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(f"Test loader ready (batch size = {batch_size})")
    return test_loader
