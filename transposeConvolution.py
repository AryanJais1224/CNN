
# 1. Imports
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 2. Create Transposed Convolution Layer
def create_transpose_conv():
    """
    Create and return a ConvTranspose2d instance.
    """
    in_channels = 3     # RGB input
    out_channels = 15   # Number of output feature maps
    kernel_size = 5     # Should typically be odd
    stride = 1
    padding = 0

    conv_transpose = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding
    )

    return conv_transpose


# 3. Inspect Layer Parameters
def inspect_layer(layer):
    """
    Print layer configuration and parameter shapes.
    """
    print(layer)
    print()
    print(f"Weight tensor shape: {layer.weight.shape}")
    print(f"Bias tensor shape:   {layer.bias.shape}")


# 4. Visualize Kernels
def visualize_kernels(layer):
    """
    Visualize kernels connecting input channel 0
    to each output channel.
    """
    fig, axs = plt.subplots(3, 5, figsize=(10, 5))

    for i, ax in enumerate(axs.flatten()):
        kernel = torch.squeeze(layer.weight[0, i, :, :]).detach()
        ax.imshow(kernel, cmap="Purples")
        ax.set_title(f"L1(0) → L2({i})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# 5. Create Random Image
def create_random_image():
    """
    Create a random RGB image tensor.
    Shape: (batch, channels, height, width)
    """
    image_size = (1, 3, 64, 64)
    img = torch.rand(image_size)
    return img


# 6. Visualize Image
def visualize_image(img):
    """
    Convert tensor (channels-first) to
    matplotlib format (channels-last).
    """
    img_to_view = img.permute(0, 2, 3, 1).numpy()

    print("Tensor shape:", img.shape)
    print("Matplotlib shape:", img_to_view.shape)

    plt.imshow(np.squeeze(img_to_view))
    plt.title("Input Image")
    plt.axis("off")
    plt.show()


# 7. Apply Transposed Convolution
def apply_transpose_conv(layer, img):
    """
    Apply transposed convolution and return result.
    """
    result = layer(img)

    print("Input shape: ", img.shape)
    print("Output shape:", result.shape)

    return result


# 8. Visualize Output Feature Maps
def visualize_feature_maps(conv_result):
    """
    Visualize each output feature map.
    """
    fig, axs = plt.subplots(3, 5, figsize=(10, 5))

    for i, ax in enumerate(axs.flatten()):
        feature_map = torch.squeeze(conv_result[0, i, :, :]).detach()
        ax.imshow(feature_map, cmap="Purples")
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

