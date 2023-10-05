#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import cv2
import string

from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.augmentations import functional as F


class PepperPatches(ImageOnlyTransform):
    """
    Apply pixel noise to the input image.
    Args:
        value ((float, float, float) or float): color value of the pixel. 
        prob (float): probability to add pixels
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """
    def __init__(self, nb_patches=(1, 5), height=(0.1, 0.8), width=(0.1, 0.8), density=(0.05, 0.1), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.density = density
        self.nb_patches = nb_patches
        self.height = height
        self.width = width
        
    def apply(self, image, **params):
        density = (self.density[1] - self.density[0]) * np.random.random_sample() + self.density[0]
        patches = self._get_patches(image.shape[:2])

        for x1, y1, x2, y2 in patches:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if np.random.random_sample() <= density:
                        image[y, x] = 0
        return image

    def _get_patches(self, image_shape):
        image_height, image_width = image_shape[:2]
        offset = int(image_height/100)
        patches = []
        for _n in range(np.random.randint(*self.nb_patches)):
            patch_height = int(image_height * np.random.uniform(*self.height))
            patch_width = int(image_width * np.random.uniform(*self.width))
            # Offset to ensure the image borders remain white
            y1 = np.random.randint(offset, image_height - patch_height - offset)
            x1 = np.random.randint(offset, image_width - patch_width - offset)
            patches.append((x1, y1, x1 + patch_width, y1 + patch_height))
        return patches

class RandomCaption(ImageOnlyTransform):
    """
    Add random caption to the image.
    """
    def __init__(self, font=(0, 5), font_scale=(1, 2), thickness=(1, 4), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def get_text(self): 
        if random.random() > 0.5:
            style = random.choice(['roman_parenthesis', 'roman_brackets', 'roman_formula', \
                                'roman_number' , 'number', 'continued', 'scheme_number'])
            roman_number = random.choice(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', \
                                        'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'])

            if style == 'number':
                return [str(random.randint(1, 20))]
            
            if style == 'roman_parenthesis':
                return ['(' + roman_number + ')']

            if style == 'roman_brackets':
                return ['[' + roman_number + ']']

            if style == 'roman_formula':
                return ['[Formula '+ roman_number + ']']

            if style == 'roman_number':
                return [roman_number]

            if style == 'scheme_number':
                return ['Scheme ' + str(random.randint(0, 20))]
            
            if style == 'continued':
                return ['-continued']
        else:
            text1 = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(int(random.random()*50)))
            text2 = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(int(random.random()*50)))
            text3 = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(int(random.random()*50)))
            return [text1, text2, text3]

    def apply(self, image, **params):
        texts = self.get_text()
        

        font = random.randint(*self.font)
        font_scale = random.randint(*self.font_scale)
        thickness = random.randint(*self.thickness)
        
        if ((font_scale == 3) or (font_scale == 4)) and (len(texts[0]) > 4) and (position[0] > (0.9*image.shape[0])):
            position = random.choice([(int(random.uniform(0.05, 0.65)*image.shape[0]), int(random.uniform(0.05, 0.20)*image.shape[0])),
                                  (int(random.uniform(0.05, 0.65)*image.shape[0]), int(random.uniform(0.80, 0.95)*image.shape[0]))])
        if len(texts) > 1:
            position = random.choice([(int(random.uniform(0.05, 0.35)*image.shape[0]), int(random.uniform(0.05, 0.10)*image.shape[0])),
                                  (int(random.uniform(0.05, 0.35)*image.shape[0]), int(random.uniform(0.80, 0.85)*image.shape[0]))])
        else:
            position = random.choice([(int(random.uniform(0.05, 0.75)*image.shape[0]), int(random.uniform(0.05, 0.30)*image.shape[0])),
                                  (int(random.uniform(0.05, 0.75)*image.shape[0]), int(random.uniform(0.80, 0.95)*image.shape[0]))])
        for text in texts:
            image = cv2.putText(image, text, position, font, font_scale, color=(0,0,0), thickness=thickness, lineType=cv2.LINE_AA)
            position = (position[0], position[1] + 50)
        return image

class RandomLines(ImageOnlyTransform):
    """
    Add random lines to the image.
    """
    def __init__(self, nb_lines=(1, 3), thickness=(3, 10), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.nb_lines = nb_lines
        self.thickness = thickness

    def apply(self, image, **params):
        nb_lines = random.randint(*self.nb_lines)
        thickness = random.randint(*self.thickness)

        for i in range(nb_lines):
            x1 = int(random.uniform(0.05, 0.95)*image.shape[0])
            y1 = int(random.uniform(0.05, 0.95)*image.shape[0])
            x2 = int(random.uniform(0.05, 0.95)*image.shape[0])
            y2 = int(random.uniform(0.05, 0.95)*image.shape[0])
            image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=thickness)
        return image

