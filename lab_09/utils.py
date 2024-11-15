from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_heatmap(image_path, heatmap, title = None):
    """
    Overlay CAM heatmap on the original image.

    Args:
        image_path (str): Path to the original image.
        heatmap (torch.Tensor): Grad-CAM heatmap.

    """
    image = Image.open(image_path).convert('RGB')

    heatmap_resized = F.interpolate(input=heatmap.unsqueeze(0).unsqueeze(0),
                                 size=(image.size[1], image.size[0]),
                                 mode="bilinear").squeeze().detach().numpy()

    #plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()