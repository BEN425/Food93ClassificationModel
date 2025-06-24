'''
cam.py

CAM relative functions
'''


import torch
# from torch.func import vmap # Not available in older version of Pytorch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Normalize CAM of each class instead of across all classes
@torch.no_grad()
def class_activation_map(model: nn.Module, image: torch.Tensor) -> np.ndarray :
    '''
    Generate CAM from an image

    Arguments :
        model `ModifiedResNet50`: CNN classification model
        image `Tensor`: Image in Tensor format, shape: [1, 3, 224, 224]

    Return :
        `np.ndarray`: CAM of each class, shape: [93, 14, 14]
        
    '''

    assert image.shape == (1, 3, 224, 224), "Input dimension should be [1, 3, 224, 224]"

    # Get the feature map of the last conv layer and the FC layer weights
    feature_map = model.convol_last_layer(image)  # shape: [1, 2048, 14, 14]
    fc_weights = model.fc.weight.data             # shape: [93, 2048]
    
    # Compute CAMs using Einstein summation
    cams = torch.einsum('oc,bcxy->boxy', fc_weights, feature_map)  # shape: [1, 93, 14, 14]
    cams = F.relu(cams)  # Keep only positive values

    # Remove batch dimension
    # cams = cams.squeeze(0)  # shape: [93, 14, 14]

    # Normalize each class (channel) separately
    # def normalize_channel(cam):
    #     min_val = cam.min()
    #     max_val = cam.max()
    #     return (cam - min_val) / (max_val - min_val + 1e-6)
    # return torch.vmap(normalize_channel, 2, 2)(cams)

    for i in range(cams.shape[0]):
        cam = cams[i]
        min_val = cam.min()
        max_val = cam.max()
        cams[i] = (cam - min_val) / (max_val - min_val + 1e-6)

    return cams.squeeze().detach().cpu().numpy()

# Show CAM and overlapped CAM
@torch.no_grad()
def show_cam(img_original: Image.Image, cam_cls: np.ndarray, original_cam = False) :
    '''
    Show CAM and overlapped CAM

    Arguments :
        img_original `PIL.Image`: Original image
        cam_cls `np.ndarray`: CAM of the class to be shown, 2x2 array
        original_cam `bool`: Show the original size CAM. Default is False
    '''

    # Resize image and convert to numpy array
    img_original = img_original.resize((224, 224))
    img_original = np.array(img_original)

    # Convert to 3 channels (BGR)
    if img_original.ndim == 2:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    elif img_original.shape[2] == 4:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGBA2BGR)
    else:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    # Resize CAM and apply color map
    cam_resized = cv2.resize(cam_cls, (224, 224))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay original image and CAM
    overlay = cv2.addWeighted(img_original, 0.5, cam_color, 0.5, 0)

    # CAM Figure

    # Plot original 14x14 CAM
    if original_cam :
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax2.imshow(cam_cls, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        ax2.axis('off')
        mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        fig.colorbar(mappable, ax=ax2, fraction=0.046, pad=0.04)
    # Plot resized and overlay image only
    else :
        fig, (ax1) = plt.subplots(1, 1, figsize=(14, 7))

    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    mappable.set_array(cam_cls)
    fig.colorbar(mappable, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Merge original image, CAMs and probability into an image
@torch.no_grad()
def merge_image_cam(
    img_original: Image.Image,
    cam: np.ndarray,
    sorted_preds: "list[tuple[float, str]]",
    pos_preds: "list[tuple[float, str]]",
    class_id: "dict[str, int]",
    show=False,
) -> Figure :
    
    '''
    Merge original image, overlay CAMs and probability into an image

    Probability contains top-5 predictions. CAMs contain positive predictions (probability >= 0.5)

    Arguments :
        img_original `PIL.Image`: Original image
        cam `np.ndarray`: CAMs of all classes, shape: [93, 14, 14]
        sorted_preds `list[tuple[float, str]]`: List of tuples containing probability and class name, sorted by probability in descending order
        pos_preds `list[tuple[float, str]]`: List of tuples containing probability and class name, with probability >= 0.5 only
        class_id `dict[str, int]`: Dict mapping class name to class id
        show `bool`: Show the image. Default is False

    Return :
        `Figure`: Matplotlib figure

    - Remember to close the figure after using it
        
    '''

    # Show CAM of highest probability if there are no positive predictions
    if len(pos_preds) == 0 and len(sorted_preds) > 0:
        pos_preds = [sorted_preds[0]]

    # Resize image and convert to numpy array
    img_resize = img_original.resize((224, 224))
    img_resize = np.array(img_resize)

    # Convert to 3 channels (BGR)
    if img_resize.ndim == 2:
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_GRAY2BGR)
    elif img_resize.shape[2] == 4:
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGBA2BGR)
    else:
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR)

    # Draw probability

    # Empty image
    prob_img = np.zeros((224, 224, 3), dtype=np.uint8)
    prob_img.fill(255)
    # Draw class and probability
    font = ImageFont.truetype("msjh.ttc", size=16)
    prob_img_pil = Image.fromarray(prob_img)
    draw = ImageDraw.Draw(prob_img_pil)
    y = 0
    for prob, cls_name in sorted_preds[:5] :
        draw.text((20, y), cls_name, font=font, fill=(0,0,0))
        draw.text((20, y+20), f"{prob:.5f}", font=font, fill=(0,0,0))
        y += 45
    # Conver to numpy array
    prob_img = np.asarray(prob_img_pil)

    # Figure
    fig = plt.figure(figsize=(14, 7))
    
    # Plot original image
    prob_img = np.concatenate((prob_img, img_resize), axis=1)
    ax1 = fig.add_subplot(1, len(pos_preds)+2, (1, 2))
    ax1.axis("off")
    ax1.imshow(cv2.cvtColor(prob_img, cv2.COLOR_BGR2RGB))
    ax1.title.set_text("Original Image")

    # Iterate through all classes
    for idx, (prob, cls_name) in enumerate(pos_preds, start=3) :
        # Resize CAM and apply color map
        cam_cls = cam[class_id[cls_name]]
        cam_resize = cv2.resize(cam_cls, (224, 224))
        cam_uint8 = (cam_resize * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Overlay original image and CAM
        overlay = cv2.addWeighted(img_resize, 0.5, cam_color, 0.5, 0)

        # Plot overlay image
        ax2 = fig.add_subplot(1, len(pos_preds)+2, idx)
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.axis('off')
        mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        mappable.set_array(cam_cls)
        fig.colorbar(mappable, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text(f"{cls_name}: {prob:.3f}")
    
    plt.tight_layout()
    if show: plt.show()
    return fig
