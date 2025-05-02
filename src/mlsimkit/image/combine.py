# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cv2


def combine_images(image1, image2, image3, border_spacing=10, resize_factor=None):
    """
    Combines three NumPy arrays into a single image, with optional border spacing and resizing.

    Args:
        image1 (np.ndarray): NumPy array for the left image
        image2 (np.ndarray): NumPy array for the middle image
        image3 (np.ndarray): NumPy array for the right image
        border_spacing (int, optional): The spacing (in pixels) between the combined images. Default is 10.
        resize_factor (float or None, optional): Factor to resize the combined image by. If None, no resizing is performed.

    Returns:
        np.ndarray: The combined image as a NumPy array.
    """
    # Check that all input images have the same shape
    if not (image1.shape == image2.shape == image3.shape):
        raise ValueError("Images must have the same shape, {image1.shape}, {image2.shape}, {image3.shape}")

    # Get the combined width and height
    height = image1.shape[0]
    width = image1.shape[1]
    combined_width = width * 3 + border_spacing * 2

    # Create a blank canvas for the combined image
    combined = np.zeros((height, combined_width, 3), dtype=np.uint8)

    # Copy the image1, image2, and image3 images to the combined image
    combined[:, :width] = image1
    combined[:, width + border_spacing : 2 * width + border_spacing] = image2
    combined[:, 2 * width + 2 * border_spacing :] = image3

    # Resize the combined image if specified
    if resize_factor is not None:
        combined = cv2.resize(
            combined, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA
        )

    return combined
