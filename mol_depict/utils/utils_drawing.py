#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import torch

# System
import os

# RDKit
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Visualization
import matplotlib.pyplot as plt
from cairosvg import svg2png
from PIL import Image

# Modules 
from mol_depict.generation.generation import RDKitGenerator
from mol_depict.generation.generation_cdk import CDKGenerator


def draw_molecule_test(smiles, path="image.png"):
    """
    Draw molecule for testing.
    """
    molecule = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(512, 512, -1, -1, False)
    options = drawer.drawOptions()
    #options.addAtomIndices = True
    #options.addBondIndices = True
    options.setAtomPalette({17: (0, 0, 0)})
    #options.bondLineWidth = 10
    
    if molecule.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(molecule)
    Chem.SanitizeMol(molecule)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    image_svg = drawer.GetDrawingText()
    svg2png(bytestring=image_svg,write_to=path,output_width=512, output_height=512)

    image = Image.open(path)

    """
    with open("image.svg", 'w') as file:
        file.write(image_svg)    
    reportlab_graphic = svg2rlg("image.svg")
    image = renderPM.drawToPILP(reportlab_graphic).convert('RGB')
    """

    return image

def draw_molecule_rdkit(smiles, molecule=None, path="image.png", augmentations=True, fake_molecule=False, display_atom_indices=False):
    generator = RDKitGenerator(
        smiles,
        molecule=molecule, 
        fake_molecule=fake_molecule, 
        image_filename_png=path, 
        augmentations=augmentations,
        display_atom_indices=display_atom_indices
    )
    image = generator.generate_image()
    if image is None:
        return None
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def draw_molecule_cdk(smiles):
    generator = CDKGenerator()
    image, masks, labels, molecule_size = generator.generate_markush_image_masks_labels(smiles,  {2: ((),)}, 0)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def draw_molecule_keypoints_rdkit(smiles, molecule=None, path="image.png", save_molecule=False, augmentations=True, fake_molecule=False, display_atom_indices=False):
    generator = RDKitGenerator(
        smiles, 
        molecule=molecule, 
        fake_molecule=fake_molecule, 
        image_filename_png=path, 
        save_molecule=save_molecule, 
        augmentations=augmentations, 
        display_atom_indices=display_atom_indices
    )
    image, keypoints = generator.generate_image_with_keypoints()
    
    if (image is None) or (keypoints is None):
        return None, None
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image, keypoints