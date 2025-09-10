import os

import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2.functional
from torchvision.transforms.v2.functional import adjust_contrast, adjust_brightness
import matplotlib.pyplot as plt
import glob
from torchvision.io import read_image, write_jpeg
import torchvision.transforms as transforms

def apply_gaussian_noise(img):
    return torchvision.transforms.v2.functional.gaussian_noise_image(img,0.0,0.07, True)

def apply_blur(img):
    blur = torchvision.transforms.GaussianBlur(kernel_size=5,sigma=(1,3))
    return blur(img)

def apply_contrast_luminosity(img):
    contrast_img = adjust_contrast(img,random.uniform(0.5, 2))
    return adjust_brightness(contrast_img,random.uniform(0.5, 2))



def random_transform(img):
    x = random.randint(1, 4)
    if x == 1:
        return img
    elif x == 2:
        return apply_gaussian_noise(img)
    elif x == 3:
        return apply_blur(img)
    elif x == 4:
        return apply_contrast_luminosity(img)

# ---- Dataset che restituisce dict ----
class DictDataset(Dataset):
    def __init__(self, images, labels, image_size=(256, 256)):
        self.images = images
        self.labels = labels
        self.transform = transforms.Resize(image_size) #necessario che tutte le immagini abbiano la stessa dimensione

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_resized = self.transform(image)
        return {
            'image': image_resized,
            'label': self.labels[idx]
        }


# ---- Funzione principale ----
def apply_transformation(dataloader):
    all_images = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image']
            labels = batch['label']
            transformed_images = []
            for img in inputs:
                transformed_img = random_transform(img)
                transformed_images.append(transformed_img)
            all_images.append(torch.stack(transformed_images))
            all_labels.append(labels)

    all_images = torch.cat(all_images)  # (N, C, H, W)
    all_labels = torch.cat(all_labels)  # (N,)

    dataset = DictDataset(all_images, all_labels)
    return DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=False)


if __name__ == "__main__":
    # Carica immagini dalla cartella "images/"
    image_paths = sorted(glob.glob("simo/images/*.jpg"))  # Ordina i percorsi per coerenza
    images = [read_image(path).float() / 255.0 for path in image_paths]
    labels = torch.arange(len(images))  # etichette fittizie (numeri della lista)

    dataset = DictDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Dataset size:", len(dataset))
    print("Loader batches:", len(dataloader))

    # Applica trasformazioni
    new_loader = apply_transformation(dataloader)

    # Prepara la cartella di output
    output_dir = "simo/images_trasformed"
    # Questa riga crea la cartella se non esiste, altrimenti non fa nulla
    os.makedirs(output_dir, exist_ok=True)

    # Salva le immagini trasformate
    image_counter = 1
    with torch.no_grad():
        for batch in new_loader:
            for img in batch['image']:
                new_filename = f"transformed_{image_counter}.jpg"
                new_path = os.path.join(output_dir, new_filename)

                # Converti l'immagine da float [0, 1] a uint8 [0, 255]
                img_uint8 = (img * 255).to(torch.uint8)

                # Salva l'immagine
                write_jpeg(img_uint8, new_path)
                print(f"Immagine salvata: {new_path}")

                image_counter += 1