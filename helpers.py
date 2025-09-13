import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

def print_section(title, char='-'):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    print(f"{char * 3} {title} {char * (width - len(title) - 6)}")


def print_title(title, char='-'):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    half_width = int((width - len(title) - 2) / 2)
    print(f"{char * half_width} {title} {char * (width - half_width - len(title) - 3)}")


#---------------------------------HEATMAP CODE-------------------------------------

def heatmap_helpers(cam, model,input_tensor,original_image):

    logits = model(input_tensor)
    predicted_class = torch.argmax(logits, dim=1).item()

    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Il risultato è una mappa 3D (batch_size, H, W). Lo trasformiamo in 2D.
    grayscale_cam = grayscale_cam[0, :]

    rgb_img_resized = np.array(original_image.resize((224, 224))).astype(np.float32) / 255
    cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)

    # Visualizza i risultati
    display_images(cam_image, original_image)
    print(f"Prediction: {'Real' if predicted_class == 0 else 'Fake'}")


def display_images(cam_image, original_image):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cam_image)
    axes[0].set_title("Heatmap")
    axes[0].axis('off')

    axes[1].imshow(original_image)
    axes[1].set_title("Original Image")
    axes[1].axis('off')

    plt.show()
