from semantic.carla_controller.labels import SEMANTIC_COLORS

import numpy as np
import os

from math import floor
from PIL import Image
from skimage.transform import resize
from tensorflow import keras
from typing import Optional, Tuple


def infer(model, img : Image) -> np.ndarray:
    '''
    infer takes a model and an image; it does all necessary
    setup and processes the image, returning a set of
    labeled pixels    
    '''
    # Get input size of network
    input_size = model.layers[0].get_output_at(0).get_shape().as_list()
    # input_size is a list with the following options:
    # [batch_size (None), width, height, channels]. Really we just want
    # width and height:
    input_size = (input_size[1], input_size[2])

    img_input = rgb_image_to_input(img, input_size=input_size)

    return model.predict(img_input)[0]


def rgb_image_to_input(img: Image, input_size : Optional[Tuple[int, int]] = None) -> np.ndarray:
    # Ensure that we have a 3 channel RGB image (RGBA breask this)
    img = img.convert("RGB")

    # We prepare our input - a (1, w, h, 3) tensor where the first 1 is our batch size
    nn_input = np.zeros((1,) + input_size + (3,), dtype="float32")
    
    # Resize image down
    if input_size is not None and img.size != input_size + (3,):
        img = img.resize(input_size)
    
    nn_input[0] = img

    return nn_input


def labels_to_image(labels : np.ndarray, output_size : Optional[Tuple[int, int]] = None) -> Image:
    # Reduce dimensionality - instead of one hot encoded pixels, do a singular dimension
    mask = np.argmax(labels, axis=-1)
    
    # This gets us to a shape of (width, height) - we want (width, height, 1)
    mask = np.expand_dims(mask, axis=-1)

    # Resize to the output size if necessary. Note that PIL expects a differently
    # ordered image, so we reverse the dimensions
    if output_size is not None and output_size != labels.shape:
        # mask = resize(mask, tuple(reversed(output_size)), order=0)
        mask = resize(mask, output_size[::-1], order=0, preserve_range=True, anti_aliasing=False)
    
    # Next we convert the resized labels to an rgb set
    img = np.zeros(mask.shape[0:2] + (3,), dtype="uint8")

    for key in SEMANTIC_COLORS.keys():
        img[np.all(mask == key, axis=-1)] = SEMANTIC_COLORS[key]

    return Image.fromarray(img)


def overlay_labels_on_input(img: Image, labels : np.ndarray, alpha : float = 0.4) -> Image:
    labels_img = labels_to_image(labels, output_size=img.size).convert('RGBA')

    return overlay_labels_img_on_input(img, labels_img, alpha)


def overlay_labels_img_on_input(img: Image, labels_img: Image, alpha : float = 0.4) -> Image:
    img = img.convert("RGBA")
    # Pillow expects alpha to be 0 (full transparency) to 255 (full opaque)
    # so convert our percentage to an integer
    alpha_setting = floor(255 * alpha)
    labels_img.putalpha(alpha_setting)

    # Let's remove all unlabeled pixels, which would otherwise darken the image as
    # they are black
    labels_img = np.array(labels_img, dtype="uint8")
    labels_img[np.all(labels_img == (0, 0, 0, alpha_setting), axis=-1)] = (0,0,0,0)
    labels_img = Image.fromarray(labels_img)

    # Return the composite
    return Image.alpha_composite(img, labels_img)


def stitch_high_res_overlays_together(model, img : Image, sub_sections : int = 4) -> Image:
    """
    Overwrite some classes with higher resolution images onto the labels image.

    :param model: model to use for inference
    :param Image img: Pillow Image
    :param int sub_sections: number of sliding windows to use for higher resolution overwrites
    :return Image label_image:
    """

    base_labels = infer(model, img)
    base_label_image = labels_to_image(base_labels, img.size)

    # Determine the width of each subsection
    width, height = img.size
    subsection_width = width/sub_sections

    # For each subsection, get a higher res label and paste it over top the base label image for certain classes
    for i in range(sub_sections):

        # Determine the crop points based on the subsection dimensions
        crop_points = (
            int(subsection_width*i),
            int((height-subsection_width)/2),
            int(subsection_width*(i+1)),
            int((height+subsection_width)/2),
        )

        # Crop the image to allow for a zoomed and higher resolution image and convert for model input
        img_cropped = img.crop(crop_points)

        # Get labels of high res subsection
        overlay_labels = infer(model, img_cropped)
        # Get image of high res output
        overlay_labels_image = labels_to_image(overlay_labels, img_cropped.size)

        # https://note.nkmk.me/en/python-pillow-composite/ -- only for images of the same size
        # The paste functions allows us to specify a box to overlay the smaller image into
        # 0 is white, 255 is black.  White pixels will allow for overwrites
        mask = np.argmax(overlay_labels, axis=-1)
        mask = np.isin(mask, [1, 6, 7]).astype(np.uint8)*255  # only update pixels where the label value is specified
        mask = resize(mask, overlay_labels_image.size[::-1], order=0, preserve_range=True, anti_aliasing=False)

        # Convert numpy mask to pillow image
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.convert("1")  # Convert to 1 channel, needed for paste below

        # Paste this overlay using the same crop points from above and save the image
        base_label_image.paste(overlay_labels_image, box=crop_points, mask=mask_img)


    return base_label_image
