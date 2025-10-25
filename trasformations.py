import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2.functional
from torchvision.transforms.v2.functional import adjust_contrast, adjust_brightness



def apply_gaussian_noise(img):
    return torchvision.transforms.v2.functional.gaussian_noise_image(img,0.0,0.7, True)

def apply_blur(img):
    blur = torchvision.transforms.GaussianBlur(kernel_size=5,sigma=(1.0,3.0))
    return blur(img)

def apply_contrast_luminosity(img):
    contrast_img = adjust_contrast(img,random.uniform(0.5, 2))
    return adjust_brightness(contrast_img,random.uniform(0.5, 2))



def random_transform(img):
    x = random.randint(1, 5)
    if x == 1 or x == 2:
        return img
    elif x == 3:
        return apply_gaussian_noise(img)
    elif x == 4:
        return apply_blur(img)
    elif x == 5:
        return apply_contrast_luminosity(img)
    return None


# Dataset that returns a dictionary.
class DictDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx]
        }


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