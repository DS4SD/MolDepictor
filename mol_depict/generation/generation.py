#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RDKit
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdmolfiles, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D

# Mathematics
import numpy as np
import random
import math

# Python Standard Library
import tempfile
import os
import logging
logger = logging.getLogger("ocsr_logger")
import uuid
import copy

# Other
import xml.etree.ElementTree as ET
from cairosvg import svg2png
from PIL import Image

# Modules
from mol_depict.utils.utils_generation import get_options_dict, coin_flip, get_rdkit_origin, rotate
from mol_depict.utils.utils_molecule import get_molecule_from_smiles, get_molecule_size, is_aromatic
from mol_depict.generation.molecule_transformation import MoleculeTransformer


class RDKitGenerator:
    def __init__(
        self, 
        smiles, 
        molecule = None, 
        fake_molecule = True, 
        random_conformation = False, 
        halogen_subtituent_organometallic_labels = None,
        labels_terminal_carbons = None, 
        image_filename_png = None, 
        save_molecule = False, 
        augmentations = True,
        remove_stereochemistry = False,
        display_atom_indices = False
    ):
        self.smiles = smiles
        self.remove_stereochemistry = remove_stereochemistry
        if molecule:
            self.molecule = molecule
        else:
            self.molecule = get_molecule_from_smiles(
                smiles, 
                remove_stereochemistry = self.remove_stereochemistry
            )  
        self.molecule_size = get_molecule_size(self.molecule)
        self.fake_molecule = fake_molecule
        self.halogen_subtituent_organometallic_labels = halogen_subtituent_organometallic_labels
        self.labels_terminal_carbons = labels_terminal_carbons
        self.image_filename_png = image_filename_png
        self.save_molecule = save_molecule
        self.augmentations = augmentations
        self.random_conformation = random_conformation
        self.output_size = (1024, 1024)
        self.display_atom_indices = display_atom_indices
        self.delete_indices = [] # tmp hack
    
    def set_default_drawing_options(self):
        #For debugging, the seed can be set.
        #self.options.fontFile = os.path.dirname(__file__) + "/../../external/rdkit/Data/Fonts/lora.ttf"
        self.options.setAtomPalette({17: (0, 0, 0)})
        self.options.rotate = 0
        self.options.comicMode = False
        #self.options.bondLineWidth = 0.5
        #self.options.maxFontSize = 40
        #self.options.minFontSize = 40 
        #self.options.additionalAtomLabelPadding = 0.0031
        #self.options.annotationFontScale = 0.5
        #self.options.addAtomIndices = False
        #self.options.explicitMethyl = False

    def set_drawing_options(self):
        """
        This function randomly set drawing options regarding the complexity of the molecule to draw.
        If the molecule has a great complexity, lower font size and bond width are prefered. 
        Parameters available are : 
            - bond width
            - font
              - font size
            - rotation
            - hand-drawing style 
            - atom padding
            - display of atoms indexes
              - padding of atoms indexes 
        """
        
        logger.debug(f"Molecule size : {self.molecule_size}")
        # Randomly set the font
        font = random.choice(["arial", "cambria", "times", "lora"]) # calibri font creates undesired straight lines
        self.options.fontFile = os.path.dirname(__file__) + "/../../external/rdkit/Data/Fonts/" + font + ".ttf"
        # Set the color the drawing color to black 
        self.options.setAtomPalette({17: (0, 0, 0)})
        # Randomly rotate the molecule
        self.options.rotate = int(random.uniform(0, 1)*360)
        # Randomly decide to use an handwritten style for bonds 
        # In particular conditions, this options lead drawer.DrawerMolecule() to not returning.
        self.options.comicMode = random.choice([False]*9 + [True]) 
        # Randomly set bond width
        bond_width = random.choice([min((1 + int(x/self.molecule_size)), 35) for x in [15, 20, 30, 50, 80, 110, 140]])
        self.options.bondLineWidth = bond_width
        # Randomly set font size
        font_size = random.choice([min((17 + int(x/self.molecule_size), 120)) for x in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800]])
        # Randomly set padding between atoms and bonds
        atom_label_padding = round(0.5/self.molecule_size, 3)
        # Randomly decide to add atoms indices
        if self.molecule_size < 10:
            self.options.annotationFontScale = random.choice([0.3, 0.4, 0.5])
            self.options.addAtomIndices = random.choice([False]*29 + [True])
        # Special parameters for small molecules
        if self.molecule_size < 10 and coin_flip(1):
            font_size = random.choice([min((17 + int(x/self.molecule_size), 180)) for x in [800, 850, 900, 950, 1000, 1050, 1100]])
            atom_label_padding = round(0.8/self.molecule_size, 3)
        self.options.maxFontSize = font_size
        self.options.minFontSize = font_size
        self.options.additionalAtomLabelPadding = min(0.4, atom_label_padding)
        self.options.explicitMethyl = random.choice([False]*9 + [True])
   

    def generate_svg_image(self,  metadata=True):
        """
        Create a SVG image for the input molecule.
        """ 
        # Debug
        #self.smiles = "C1CC1C(C#N)NC(=O)C2=CC(=CC(=C2)Br)[N+](=O)[O-]"
        #print(self.smiles)
        if self.remove_stereochemistry == False and (("@@" in self.smiles) or ("@" in self.smiles) or ("/" in self.smiles)):
            stereochemistry = True
        else:
            stereochemistry = False
 
        if self.augmentations and stereochemistry:
            # Add explicit hydrogens (with stereo-chemistry)
            # Recomputing 2D coordinates will generate a new conformation, and augment the molecule
            self.molecule = Chem.AddHs(self.molecule, explicitOnly=True)
            rdDepictor.Compute2DCoords(self.molecule)
   
        if stereochemistry:
            if not self.augmentations:
                self.rendering_size = (256, 256)
            else:
                if self.molecule_size > 10:
                    self.rendering_size = (256, 256)
                else:
                    self.rendering_size = random.choice([(128, 128), (256, 256)])
        else:
            self.rendering_size = (1024, 1024)
        drawer = rdMolDraw2D.MolDraw2DSVG(self.rendering_size[0], self.rendering_size[1], -1, -1, False) 
        self.options = drawer.drawOptions()

        logger.debug("Set drawing options")
        # Set drawing options
        if not(self.augmentations):
            self.set_default_drawing_options()
            
        elif self.augmentations and (not stereochemistry):
            # TODO Adjust settings to allow molecules with stereochemistry to be augmented (different rendering size)
            self.set_drawing_options()
         
        elif self.display_atom_indices:
            self.options.addAtomIndices = True
            self.options.setAtomPalette({17: (0, 0, 0)})
        else:
            self.options.setAtomPalette({17: (0, 0, 0)})
            
            #if not stereochemistry:
            #    self.options.bondLineWidth = 10
            #    self.options.maxFontSize = 45
            #    self.options.minFontSize = 45

        logger.debug("Modify molecule")
        # Randomly transform the molecule 
        if not hasattr(self, 'matches_indices'):
            self.matches_indices = []
        
        if self.augmentations and (not stereochemistry) and (not self.options.comicMode) and (not self.options.minFontSize > 140):
            self.molecule = MoleculeTransformer(self.molecule, self.molecule_size, self.options, self.matches_indices, self.fake_molecule, self.random_conformation).transform()
        
        # Draw stereo any bonds as wavy
        display_wavy_bond = False
        display_wavy_bond_proportion = 0.15
        if self.augmentations and coin_flip(display_wavy_bond_proportion):
            logger.debug("Display wavy bond")
            for bond in self.molecule.GetBonds():
                # Select a single bond to transform to double
                if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE) and (bond.GetStereo() == 0) and (bond.GetIsAromatic() == False):
                    match = True
                    # Verify that neighboring atoms are simple carbons
                    for atom_index in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                        atom = self.molecule.GetAtomWithIdx(atom_index)
                        if (atom.GetNumImplicitHs() <= 1) or (atom.GetSymbol() != "C") or \
                                (atom.GetFormalCharge() != 0) or atom.IsInRing() or atom.HasProp("_displayLabel"):
                            match = False
                            break
                    
                    if not match:
                        continue

                    # Verify that neighboring bonds are single bonds, and only one at each side
                    begin_neigbors_atoms = [atom_index.GetIdx() for atom_index in self.molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetNeighbors() if atom_index.GetIdx() != bond.GetEndAtomIdx()]
                    end_neigbors_atoms = [atom_index.GetIdx() for atom_index in self.molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetNeighbors() if atom_index.GetIdx() != bond.GetBeginAtomIdx()]
                    if len(begin_neigbors_atoms) > 1 or len(end_neigbors_atoms) > 1:
                        continue

                    for begin_neigbor_atom in begin_neigbors_atoms:
                        neigboring_bond = self.molecule.GetBondBetweenAtoms(begin_neigbor_atom, bond.GetBeginAtomIdx())
                        if neigboring_bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                            match = False
                    
                    for end_neigbor_atom in end_neigbors_atoms:
                        neigboring_bond = self.molecule.GetBondBetweenAtoms(end_neigbor_atom, bond.GetEndAtomIdx())
                        if neigboring_bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                            match = False

                    if not match:
                        continue

                    bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                    bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY) 

                    # Set wavy bonds around double bonds with STEREOANY stereo
                    rdmolops.AddWavyBondsForStereoAny(self.molecule, addWhenImpossible=0, clearDoubleBondFlags=False) 

                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE) 
                    display_wavy_bond = True
                    break
        
        # Select randomly the library used to generate molecule coordinates
        # CoordGen can currently generate segmentation fault.
        #Chem.rdDepictor.SetPreferCoordGen(random.choice([False,True]))
        if not self.molecule.GetNumConformers():
            rdDepictor.Compute2DCoords(self.molecule)

        # MoleculeTransformer modify some of the drawing parameters
        logger.debug(f"Drawing options : {get_options_dict(self.options)}") 
        logger.debug("Draw molecule")
        drawer.DrawMolecule(self.molecule)

        if metadata:
            # Compute 2D coords destroys conformations 
            #rdDepictor.Compute2DCoords(self.molecule)
            drawer.AddMoleculeMetadata(self.molecule)

        drawer.FinishDrawing()
        self.image_svg = drawer.GetDrawingText()

        display_aromatic_cycles_proportion = 0.15 # Adding aromatic cycles 
        self.display_aromatic_cycles = False
        
        ringInfo = self.molecule.GetRingInfo()
        atomRings = ringInfo.AtomRings()
        if not display_wavy_bond and (len(atomRings) > 0) and coin_flip(display_aromatic_cycles_proportion) and self.augmentations:
            logger.debug("Add aromatic cycles")
            self.display_aromatic_cycles = self.add_aromatic_cycles()
            rdmolops.SetAromaticity(self.molecule) # Automatically set by MolToMolFile()
            return self.image_svg
        
        return self.image_svg

    def generate_image(self):
        # Generate SVG image
        self.generate_svg_image(metadata=False)
        
        tmp = False
        if not self.image_filename_png:
            tmp = True
            image_file, self.image_filename_png = tempfile.mkstemp()
        """
        # Debugging
        print(self.image_svg)
        svg2png(
            bytestring=self.image_svg, 
            write_to=self.image_filename_png, 
            output_width=self.output_size[0], 
            output_height=self.output_size[1]
        )
        """
        try:
            #print(self.image_svg)
            # Convert SVG to PNG
            svg2png(
                bytestring=self.image_svg, 
                write_to=self.image_filename_png, 
                output_width=self.output_size[0], 
                output_height=self.output_size[1]
            )
            image = Image.open(self.image_filename_png)

            if tmp:
                # Clean up files
                os.close(image_file)
                os.remove(self.image_filename_png)

        except:
            logger.error("Critic error in image PNG rendering")

            if tmp:
                #Clean up file
                os.close(image_file)
                os.remove(self.image_filename_png)
            return None

        image = np.array(image, dtype=np.float32)

        # Check if the image contains (perfectly) red pixels
        #if np.any(np.all(image == (255, 0, 0), axis=2)):
        #    logger.error("Image sanitization error : the image contains red pixels")
        #    return None

        # Images are not exactly black and white
        # Turn any pixel that is not perfectly white to a black pixel
        image[np.any(image <= (254., 254., 254.), axis=2)] = [0., 0., 0.] 
        image = (image != 0).astype(np.float32)
        return image 

    def generate_masks(self, molecule, image_svg, matches, bonds=None):
        """
        Modify the input SVG image to extract substructures masks
        Note:
            Bonds with hetero atoms are colored and correspond to 2 svg items (one for each half-bond)
        """
        self.masks_svg_list = []
        for match_index, match in enumerate(matches):  
            if bonds is None:   
                bonds = []
                molecule_editable = Chem.EditableMol(molecule)
                # Works because indices are sorted ! 
                for index in range(molecule.GetNumAtoms()-1, -1, -1):
                    if index not in match:
                        molecule_editable.RemoveAtom(index)

                molecule_cropped = molecule_editable.GetMol()
                for bond in molecule_cropped.GetBonds():
                    bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                    bonds.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

                #print(bonds)
                #draw_molecule(molecule_cropped, self.output_size[0], self.output_size[1], path=str(match_index) + ".png", display_indices=True)
            
            root = ET.fromstring(image_svg)       
            delete_children = []
            background_color = "#FFFFFF"
            root[0].set('style', 'opacity:1.0;fill:'+background_color+';stroke:none')
            for child in root:
                classes = child.get('class')
                if classes:     
                    if classes =='note':
                        # Displayed atoms indices
                        delete_children.append(child)
                        continue
                    atoms_bonds = [int(class_.split("-")[-1]) for class_ in classes.split() if 'atom' in class_]
                    if len(atoms_bonds) == 1:
                        if atoms_bonds[0] not in match:
                            delete_children.append(child)
                            continue

                    elif len(atoms_bonds) == 2:  
                        try:
                            potential_bond = (match.index(atoms_bonds[0]) , match.index(atoms_bonds[1]))
                        except ValueError:
                            potential_bond = (None, None)                    
                        if (potential_bond not in bonds):
                            delete_children.append(child)
                            continue   
                                
                    # Color matched bonds and atoms in white
                    color = "#000000"
                    child.set("stroke", color)
                    child.set("fill", color)
                    style = child.get('style')
                    if style:
                        if "fill:" in style:
                            start_index_fill = style.index("fill:") + 4 + 1
                            end_index_fill = start_index_fill + 7
                            style = style[:start_index_fill] + color + style[end_index_fill:]
                        if "stroke:" in style:
                            start_index_stroke = style.index("stroke:") + 6 + 1
                            end_index_stroke = start_index_stroke + 7
                            style = style[:start_index_stroke] + color + style[end_index_stroke:]
                        style = style.replace("evenodd", "nonzero")
                        child.set('style', style) 
                else:
                    # Black background
                    child.set("fill", background_color)
                    style = child.get('style')
                    if style:
                        if "fill:" in style:
                            start_index_fill = style.index("fill:") + 4 + 1
                            end_index_fill = start_index_fill + 7
                            style = style[:start_index_fill] + background_color + style[end_index_fill:]
                        if "stroke:" in style:
                            start_index_stroke = style.index("stroke:") + 6 + 1
                            end_index_stroke = start_index_stroke + 7
                            style = style[:start_index_stroke] + background_color + style[end_index_stroke:]
                        style = style.replace("evenodd", "nonzero")
                        child.set('style', style) 
            for child in delete_children:
                root.remove(child) 

            self.masks_svg_list.append(ET.tostring(root).decode('utf-8'))

    def generate_image_masks_labels(self, labels_bonds, image_submolecules_matches):
        """
        Given a SMILES, the function create the molecule image with random display settings and extract substructures masks.

            returns - masks, a list of numpy arrays [H, W] with boolean values (uint8)
                    - image, a numpy array [H, W, C] with boolean values (float32)
                    - labels, a list of int
        """
        labels = []
        masks = []
        Chem.rdDepictor.SetPreferCoordGen(False)

        if self.molecule_size < 1:
            logger.error("Molecule size is too small")
            return None
        
        self.matches_indices = list(np.unique([index for submolecule_matches in image_submolecules_matches.values() for match in submolecule_matches for index in match]))

        image = self.generate_image()

        logger.debug("Generate SVG for masks")
        masks = []
        labels = []

        Chem.rdDepictor.SetPreferCoordGen(False)

        for label in image_submolecules_matches.keys():

            matches = image_submolecules_matches[label]

            # Add H to matched patterns, these matches can't be precomputed because hydrogens are added/removed by some transformations and their indices modified    
            matches_list = [list(match) for match in matches]
            molecule_atoms_indices = [atom.GetIdx() for atom in self.molecule.GetAtoms()]

            '''
            Functionnal group terminal nodes are carbons or end atoms. "Then, extremities can't be connected unintentionnaly to H atoms." 
            Some groups, as halogen substituent, are specials. For example "PF" should be matched without H atoms in [PH]F, [PH2]F...
            '''
            if label not in self.halogen_subtituent_organometallic_labels:
                matches_list_with_h = copy.deepcopy(matches_list)
                for match_index, match in enumerate(matches_list):
                    for position, atom_index in enumerate(match):
                        if position not in self.labels_terminal_carbons[label]:
                            if atom_index in molecule_atoms_indices:
                                atom = self.molecule.GetAtomWithIdx(atom_index)
                                if atom.GetSymbol() in ['P', 'B', 'N', 'O', 'S', 'C']: 
                                    for neighbor in atom.GetNeighbors():
                                        if neighbor.GetSymbol() == 'H':
                                            matches_list_with_h[match_index].append(neighbor.GetIdx())
                matches = tuple([tuple(match) for match in matches_list_with_h])

            if matches != ():
                logger.debug(f"Matches : {matches}")

            # Generate SVG masks self,, bonds=None
            self.generate_masks(self.molecule, self.image_svg, matches, bonds=labels_bonds[label])  

            for mask_index, mask_svg in enumerate(self.masks_svg_list):
                if mask_index not in self.delete_indices:
                    mask_file , mask_filename_png = tempfile.mkstemp()

                    # Convert SVG to PNG
                    svg2png(
                        bytestring=mask_svg, 
                        write_to=mask_filename_png, 
                        output_width=self.output_size[0], 
                        output_height=self.output_size[1]
                    )
                    mask = Image.open(mask_filename_png)

                    # Clean up files
                    os.close(mask_file)
                    os.remove(mask_filename_png)

                    mask = np.asarray(mask)[:, :, 0]

                    # Binarize mask
                    mask = (mask > 0.1).astype(np.uint8)

                    # Append mask and label
                    masks.append(mask)
                    labels.append(label)

        if len(labels) == 0:
            logging.info("The image contains no objects")
            return None

        return image, masks, labels, self.molecule_size

    def get_keypoints(self):
        keypoints = []
        rdkit_keypoints = []

        scale_factor = self.output_size[0]/self.rendering_size[0]
        atoms = []
        bonds = []
        # Parse SVG
        root = ET.fromstring(self.image_svg)  
        for child in root:
            if child.tag == "{http://www.w3.org/2000/svg}metadata":
                molecule_metadata = child[0]
                for item in molecule_metadata:
                    if item.tag == "{http://www.rdkit.org/xml}atom":
                        atom = {
                            "idx": item.get("idx"),
                            "atom-smiles": item.get("atom-smiles"),
                            "position": (float(item.get("drawing-x")), float(item.get("drawing-y"))),
                            "rdkit_position": (float(item.get("x"))*scale_factor + scale_factor//2, float(item.get("y"))*scale_factor + scale_factor//2)
                        }
                        atoms.append(atom)
                        keypoints.append((float(item.get("drawing-x"))*scale_factor + scale_factor//2, float(item.get("drawing-y"))*scale_factor + scale_factor//2))
                        rdkit_keypoints.append((float(item.get("x"))*scale_factor + scale_factor//2, float(item.get("y"))*scale_factor + scale_factor//2))

                    if item.tag == "{http://www.rdkit.org/xml}bond":
                        bond = {
                            "idx": item.get("idx"),
                            "bond-smiles": item.get("bond-smiles"),
                            "begin-atom-idx" : item.get("begin-atom-idx"),
                            "end-atom-idx": item.get("end-atom-idx")
                        }
                        bonds.append(bond)
        
        return keypoints, rdkit_keypoints

    def generate_image_with_keypoints(self):
        # Generate SVG image
        self.generate_svg_image()
        
        tmp = False
        if not self.image_filename_png:
            tmp = True
            image_file, self.image_filename_png = tempfile.mkstemp()
        
        try:
            #print(self.image_svg)
            # Convert SVG to PNG
            svg2png(
                bytestring=self.image_svg, 
                write_to=self.image_filename_png, 
                output_width=self.output_size[0], 
                output_height=self.output_size[1]
            )
            image = Image.open(self.image_filename_png)

            # Save molecule
            if self.save_molecule:
                if self.display_aromatic_cycles:
                    rdmolfiles.MolToMolFile(
                        self.molecule, 
                        self.image_filename_png[:-4] + ".mol", 
                        kekulize = False # call setAromaticity()
                    )
                else:
                    rdmolfiles.MolToMolFile(
                        self.molecule, 
                        self.image_filename_png[:-4] + ".mol", 
                        kekulize = True 
                    )

            if tmp:
                # Clean up files
                os.close(image_file)
                os.remove(self.image_filename_png)

        except:
            logger.error("Error in image SVG to PNG rendering")
            if tmp:
                #Clean up file
                os.close(image_file)
                os.remove(self.image_filename_png)
            return None, None

        image = np.array(image, dtype=np.float32)
        
        # Check if the image contains (perfectly) red pixels
        if np.any(np.all(image == (255, 0, 0), axis=2)):
            logger.error("Image sanitization error : the image contains red pixels")
            return None, None

        # Check that no part of the molecule is overlaping with the border
        if (np.sum(image[0, :, :] != 255)) or (np.sum(image[self.output_size[0] - 1, :, :] != 255)) or (np.sum(image[:, 0, :] != 255)) or (np.sum(image[:, self.output_size[1] - 1, :] != 255)):
            logger.error("Image sanitization error : the depicted molecule overlaps with the border")
            return None, None

        # Images are not exactly black and white
        # Turn any pixel that is not perfectly white to a black pixel
        image[np.any(image <= (254., 254., 254.), axis=2)] = [0., 0., 0.] 
        image = (image != 0).astype(np.float32)

        # Keypoints
        keypoints, rdkit_keypoints = self.get_keypoints()
        rdkit_origin = get_rdkit_origin(keypoints[0], rdkit_keypoints[0], keypoints[1], rdkit_keypoints[1])
        if rdkit_origin == (None, None):
            return None, None
            
        keypoints = [rotate(rdkit_origin, point, math.radians(self.options.rotate)) for point in keypoints]
        # Images are ordered as arrays with the y axis pointing down. 
        # keypoints are not expressed on the same basis.
        
        return image, keypoints

    def add_aromatic_cycles(self):
        """
        Depict aromatic cycles with circles instead of simple and double bonds.
        """
        display_aromatic_cycles = False
        ringInfo = self.molecule.GetRingInfo()
        atomRings = ringInfo.AtomRings()
        aromBondRings = []
        aromAtomRings = []
        aromBonds = {}
        for iRing, bondRing in enumerate(ringInfo.BondRings()):
            if is_aromatic(self.molecule, bondRing):
                aromBondRings.append(bondRing)
                aromAtomRings.append(atomRings[iRing])
                for b in bondRing:
                    aromBonds[f'bond-{b}'] = []
        
        root = ET.fromstring(self.image_svg)
        
        for child in root:
            cls = child.get('class')
            d = child.get('d')
            if cls and d:
                scls = cls.split()
                if 'bond' in scls[0]:
                    bond = scls[0]
                    ds = d.replace('M','').replace('L',',').replace(' ','').split(',')
                    try:
                        beg = [float(x) for x in ds[0:2]]
                    except:
                        print("ERROR in add aromatic cycle")
                        print("Add aromatic cycles Error")
                    end = [float(x) for x in ds[2:4]]
                    dist = (end[0] - beg[0])**2+(end[1] - beg[1])**2
                    centerX = 0.5*(end[0]+beg[0])
                    centerY = 0.5*(end[1]+beg[1])
                    atomIdx = scls[1:3]
                    if bond in aromBonds:
                        aromBonds[bond].append({ 
                            'dist': dist,
                            'child': child,
                            'centerX': centerX,
                            'centerY': centerY,
                            'atomIdx': atomIdx,
                            'beg': beg,
                            'end': end
                        })
            
        for bond, data in aromBonds.items():
            if len(data) >= 2:
                root.remove(data[1]['child'])
                data.pop(1)
        
        # Find center of aromatic rings
        for aromBondRing, aromAtomRing in zip(aromBondRings, aromAtomRings):
            
            centerRingX = 0.0
            centerRingY = 0.0

            try:
                # Extent bond in case of heteroatoms
                for aromBond1 in aromBondRing:
                    atomIdx1 = aromBonds[f'bond-{aromBond1}'][0]['atomIdx']
                    beg1 = np.array(aromBonds[f'bond-{aromBond1}'][0]['beg'])
                    end1 = np.array(aromBonds[f'bond-{aromBond1}'][0]['end'])
                    for aromBond2 in aromBondRing:
                        if aromBond2 <= aromBond1: 
                            continue
                        atomIdx2 = aromBonds[f'bond-{aromBond2}'][0]['atomIdx']
                        beg2 = np.array(aromBonds[f'bond-{aromBond2}'][0]['beg'])
                        end2 = np.array(aromBonds[f'bond-{aromBond2}'][0]['end'])
                        # Solve for intersection
                        a = (np.array([-end1 + beg1, end2 - beg2])).transpose()
                        b = beg1-beg2
                        x = np.linalg.solve(a, b)
                        # Get intersection
                        intersect = x[0]*(end1-beg1) + beg1
                        # Replace intersection
                        if atomIdx1[0] == atomIdx2[0]:
                            aromBonds[f'bond-{aromBond1}'][0]['beg'] = intersect.tolist()
                            aromBonds[f'bond-{aromBond2}'][0]['beg'] = intersect.tolist()
                        elif atomIdx1[0] == atomIdx2[1]:
                            aromBonds[f'bond-{aromBond1}'][0]['beg'] = intersect.tolist()
                            aromBonds[f'bond-{aromBond2}'][0]['end'] = intersect.tolist()
                        elif atomIdx1[1] == atomIdx2[0]:
                            aromBonds[f'bond-{aromBond1}'][0]['end'] = intersect.tolist()
                            aromBonds[f'bond-{aromBond2}'][0]['beg'] = intersect.tolist()
                        elif atomIdx1[1] == atomIdx2[1]:
                            aromBonds[f'bond-{aromBond1}'][0]['end'] = intersect.tolist()
                            aromBonds[f'bond-{aromBond2}'][0]['end'] = intersect.tolist()
            except:
                print("Add aromatic cycles Warning")
                pass

            # Find center of circle
            nAromBondRing = len(aromBondRing)
            for aromBond in aromBondRing:
                atom1 = aromBonds[f'bond-{aromBond}'][0]['beg']
                atom2 = aromBonds[f'bond-{aromBond}'][0]['end']
                centerRingX += atom1[0]
                centerRingY += atom1[1]
                centerRingX += atom2[0]
                centerRingY += atom2[1]      
            centerRingX = centerRingX/float(nAromBondRing)/2.0
            centerRingY = centerRingY/float(nAromBondRing)/2.0

            # Find radius
            radius = 0.0
            for aromBond in aromBondRing:
                atom1 = aromBonds[f'bond-{aromBond}'][0]['beg']
                atom2 = aromBonds[f'bond-{aromBond}'][0]['end']
                centerX = 0.5*(atom1[0] + atom2[0])
                centerY = 0.5*(atom1[1] + atom2[1])
                radius += math.sqrt((centerRingX - centerX)**2 + (centerRingY - centerY)**2)

            radius_size = random.choice([1.4, 1.5, 1.6])
            radius = radius/(float(nAromBondRing)*radius_size)

            new_child = ET.SubElement(root, 'ns0:circle')
            new_child.set('cx', "{:.2f}".format(centerRingX))
            new_child.set('cy', "{:.2f}".format(centerRingY))
            new_child.set('r', "{:.2f}".format(radius))
            new_child.set('style', f'fill:none;stroke:#000000;stroke-width:{self.options.bondLineWidth}px' )
            new_child.set('style', f'fill:none;stroke:#000000;stroke-width:{self.options.bondLineWidth}px' )
            new_child.set('class',' '.join([ f'atom-{atom}' for atom in aromAtomRing]))
            display_aromatic_cycles = True

        self.image_svg = ET.tostring(root).decode('utf-8')
        return display_aromatic_cycles
        
