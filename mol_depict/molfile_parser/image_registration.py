#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Image
import cv2
from PIL import Image, ImageOps

# Maths
import numpy as np


def get_affine_transformation(image_rdkit, image_gt):
    # Enlarge masks
    image_rdkit = cv2.bitwise_not(image_rdkit)
    image_gt = cv2.bitwise_not(image_gt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_rdkit = cv2.dilate(image_rdkit, kernel, iterations=3)
    image_gt = cv2.dilate(image_gt, kernel, iterations=3)
    image_rdkit = cv2.bitwise_not(image_rdkit)
    image_gt = cv2.bitwise_not(image_gt)

    # Add margin
    margin = 750
    image_gt = cv2.copyMakeBorder(image_gt, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    image_rdkit = cv2.copyMakeBorder(image_rdkit, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Convert to grayscale
    image_rdkit = cv2.cvtColor(image_rdkit, cv2.COLOR_BGR2GRAY)
    image_gt = cv2.cvtColor(image_gt, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(15000)

    # Find keypoints and descriptors
    kp1, d1 = orb_detector.detectAndCompute(image_rdkit, None)
    kp2, d2 = orb_detector.detectAndCompute(image_gt, None)

    # Match features between the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = list(matcher.match(d1, d2))
    matches.sort(key = lambda x: x.distance)
    
    # Take a top % matches forward
    matches = matches[:int(len(matches)*0.8)]
    no_of_matches = len(matches)

    scores = [match.distance for match in matches] # The lower, the better
    #print("Average score: ", round(np.mean(scores), 2))
    if np.mean(scores) > 25:
        print("The predicted registration is likely to be incorrect (The average distance between keypoints is superior to 25)")
        return None, None

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Get the transformation
    transformation, _ = cv2.estimateAffinePartial2D(p1, p2, cv2.RANSAC, confidence=0.995)

    return transformation, margin