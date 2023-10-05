#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Mathematics
from numpy.random import permutation
import random

# Python Standard Library
import os
import logging
logger = logging.getLogger("ocsr_logger")

# Modules
from mol_depict.utils.utils_generation import get_abbreviations, get_options_dict, coin_flip
from mol_depict.utils.utils_molecule import get_molecule_from_smiles, get_molecule_size

PROJECT_PATH = os.path.dirname(__file__) + "/../../"


class MoleculeTransformer:
    """
    AllChem.EmbedMolecule(molecule, useRandomCoords=True) for totally random molecule projection
    
    Wavy bonds for Markush images:
        Chem.AddWavyBondsForStereoAny(self.molecule)
    """
    def __init__(self, molecule, molecule_size, options, matches_indices, fake_molecule=False, random_conformation=False):
        self.molecule = molecule
        self.molecule_size = molecule_size
        # options is a pointer
        self.options = options
        self.matches_indices = matches_indices
        self.fake_molecule = fake_molecule
        self.random_conformation = random_conformation
        self.abbreviations = get_abbreviations()

    def select_random_conformer(self):
        self.molecule = Chem.AddHs(self.molecule)
        
        # Create conformers 
        #RDLogger.DisableLog('rdApp.*')  
        AllChem.EmbedMultipleConfs(
            self.molecule, 
            numConfs = 5, 
            maxAttempts = 20, 
            pruneRmsThresh = 1, 
            useExpTorsionAnglePrefs = True,
            useBasicKnowledge = True, 
            enforceChirality = False, 
            forceTol = 0.2, 
            numThreads = 1
        )    
        #RDLogger.EnableLog('rdApp.*') 
        self.molecule = Chem.RemoveHs(self.molecule) 

        # Add bonds between P, B, N, O, S and H (only to calculate conformer coordinates, they are then removed)
        h_atoms = []
        for atom_index, atom in enumerate(self.molecule.GetAtoms()):
            if atom.GetSymbol() in ['P','B','N','O','S']: 
                h_atoms.append(atom_index)
        self.molecule = Chem.AddHs(self.molecule, onlyOnAtoms=h_atoms)

        # Select one conformer
        if len(self.molecule.GetConformers()):
            conformer_index = random.choice(range(len(self.molecule.GetConformers())))
            _ = rdDepictor.GenerateDepictionMatching3DStructure(
                self.molecule, 
                self.molecule, 
                confId = conformer_index, 
                #acceptFailure = True #TO DOUBLE CHECK! 
            )  
            
        self.molecule = Chem.RemoveHs(self.molecule)  

        if not(len(self.molecule.GetConformers())):
            logger.info("No valid conformers found")
            # Avoid letting the molecule in a broken state (for drawer.AddMetaData())
            rdDepictor.Compute2DCoords(self.molecule)

    def add_explicit_h(self):
        h_atoms = []
        for atom_index, atom in enumerate(self.molecule.GetAtoms()):
            if atom.GetSymbol() in ['P','B','N','O','S']: 
                h_atoms.append(atom_index)
        if h_atoms:      
            # AddHs add H in the 3D structure
            self.molecule = Chem.AddHs(self.molecule, onlyOnAtoms=h_atoms)
            rdDepictor.Compute2DCoords(self.molecule)
            
    def add_explicit_h_c(self):
        rings_information = self.molecule.GetRingInfo()
        rings_indices = [index for ring in rings_information.AtomRings() for index in ring]
            
        explicit_h_proportion = 0.70
        h_atoms = []
        for atom_index, atom in enumerate(self.molecule.GetAtoms()):
            if (atom.GetSymbol() == 'C') and (atom.GetTotalNumHs() == 1) and (atom.GetIdx() not in rings_indices):
                if coin_flip(explicit_h_proportion):
                    h_atoms.append(atom_index)
        if h_atoms:      
            # AddHs add H in the 3D structure
            self.molecule = Chem.AddHs(self.molecule, onlyOnAtoms=h_atoms)
            rdDepictor.Compute2DCoords(self.molecule)

    def reduce_h_bonds_size(self):
        self.molecule = rdMolDraw2D.PrepareMolForDrawing(self.molecule, addChiralHs=False)
        
        if self.molecule_size <= 7:
            bond_size_factor = random.choice([0.5, 0.6, 0.7])
        else:
            bond_size_factor = random.choice([0.8, 0.9])
        self.options.additionalAtomLabelPadding = 0
        current_size = 0
        for atom_index, atom in enumerate(self.molecule.GetAtoms()):
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':     
                    for iconf, conf in enumerate(self.molecule.GetConformers()):
                        try:
                            if current_size == 0:
                                current_size = rdMolTransforms.GetBondLength(self.molecule.GetConformer(iconf), neighbor.GetIdx(), atom_index)
                            bond_size = max(round(current_size*bond_size_factor, 3), 0.01)
                            rdMolTransforms.SetBondLength(self.molecule.GetConformer(iconf), neighbor.GetIdx(), atom_index, bond_size)
                            logger.debug(f"Reduced hydrogen bond length: {bond_size}")
                        except:
                            logger.error("RDkit transformation error: Can't modify bond length, both atoms have same coordinates.")
                            pass
              
    def remove_h_from_abbreviations(self):
        for atom in self.molecule.GetAtoms():
            if (atom.GetFormalCharge() == 0) and ((atom.GetSymbol() == 'P') or (atom.GetSymbol() == 'B') or \
                                                  (atom.GetSymbol() == 'N') or (atom.GetSymbol() == 'S') or (atom.GetSymbol() == 'O')) :
                atom.SetProp("_displayLabel", atom.GetSymbol()) 

    def add_circled_charges(self, charge_size):
        # Font size limited to lessen overlaps
        font_size = random.choice([min((17 + int(x/self.molecule_size), 110)) for x in [100, 150, 200, 250, 300]])
        self.options.maxFontSize = font_size
        self.options.minFontSize = font_size

        # Large or small circled cycles
        if charge_size == "large":
            self.options.fontFile = PROJECT_PATH + "/external/rdkit/Data/Fonts/circled_large.ttf"
            self.spacing_number = random.choice([3, 4, 5])
        if charge_size == "small":
            self.options.fontFile = PROJECT_PATH + "/external/rdkit/Data/Fonts/circled.ttf"
            self.spacing_number = random.choice([1, 2])
        spacing = " "*self.spacing_number

        for atom in self.molecule.GetAtoms():
            charge = atom.GetFormalCharge()
            if (charge == 1) or (charge == -1):
                nb_hatoms = atom.GetTotalNumHs()
                label = atom.GetSymbol()
                charge_string = "-" if  (charge == -1) else "+"
                hatoms_string = "H"*(nb_hatoms >= 1) + ("<sub>" + str(nb_hatoms) + "</sub>")*(nb_hatoms >= 2)
                atom.SetProp("_displayLabel",label+hatoms_string+"<sup>"+spacing+charge_string+"</sup>")
                atom.SetProp("_displayLabelW","<sup>"+charge_string+spacing+"</sup>"+label+hatoms_string)

    def move_charges_to_left_side(self):
        for atom in self.molecule.GetAtoms():
            charge = atom.GetFormalCharge()
            if (charge == 1) or (charge == -1):
                label = atom.GetSymbol()
                charge_string = "-" if  (charge == -1) else "+"
                atom.SetProp("_displayLabel","<sup>"+charge_string+"</sup>"+label)
                atom.SetProp("_displayLabelW","<sup>"+charge_string+"</sup>"+label)

    def add_abbreviations_extension(self, abbreviation_size, circled_charges_added):
        """
        Idea: Also set _displayLabelW
        """
        if circled_charges_added:
            # Abbreviations such as t-Bu or COO- contain charges
            return

        nb_atoms = self.molecule.GetNumAtoms()
        for index in permutation(len(self.molecule.GetAtoms())):
            index = int(index)
            atom = self.molecule.GetAtomWithIdx(index)
            if (index not in self.matches_indices) and (atom.GetTotalNumHs() > 1) and (atom.GetNumExplicitHs() == 0) and (atom.GetSymbol() == "C"):
                if abbreviation_size == "single_attachment":
                    abbreviation = random.choice(self.abbreviations["1"])
                if abbreviation_size == "multiple_attachments":
                    nb_attachments = random.choice([2]*4 + [3])
                    abbreviation = random.choice(self.abbreviations[str(nb_attachments)])
                logger.debug(f"Selected abbreviation: {abbreviation}")
                
                """
                # Add spacing for circles charges (Invalid for t-Bu)
                if circled_charges_added:
                    spacing = " "*self.spacing_number
                    if abbreviation[0] in ["-", "+"]:
                        abbreviation.replace("+", "+" + spacing)
                        abbreviation.replace("-", "-" + spacing)
                    elif abbreviation[-1] in ["-", "+"]:
                        abbreviation.replace("+", spacing + "+")
                        abbreviation.replace("-", spacing + "-")
                """

                if abbreviation_size == "single_attachment":
                    extension = Chem.MolFromSmiles('*')   
                    
                if abbreviation_size == "multiple_attachments":
                    if nb_attachments == 2:
                        extension = Chem.MolFromSmiles('*C')  

                    elif nb_attachments == 3:
                        extension = Chem.MolFromSmiles('*(C)C')  

                extension.GetAtomWithIdx(0).SetProp("_displayLabel", abbreviation)

                # Set alias for MolScribe
                alias = abbreviation.replace("<sub>", "").replace("</sub>", "")
                Chem.SetAtomAlias(extension.GetAtomWithIdx(0), alias)

                if len(abbreviation) > 4:
                    # For large abbreviation, lower the font size
                    font_size = random.choice([min((17 + int(x/self.molecule_size), 90)) for x in [100, 150, 200, 250, 300, 350, 400, 450, 500]])
                    self.options.maxFontSize = font_size
                    self.options.minFontSize = font_size
                    
                self.molecule = Chem.CombineMols(self.molecule, extension)
                self.molecule = Chem.EditableMol(self.molecule)
                self.molecule.AddBond(index, nb_atoms, order=Chem.rdchem.BondType.SINGLE)
                self.molecule = self.molecule.GetMol()
                rdDepictor.Compute2DCoords(self.molecule)
                #Chem.SanitizeMol(self.molecule) # Not sure?
                break
    
    def angle_carbons_with_triple_bond(self):
        """
        Replace triple bonds with double bonds. 
        Regarding implicit hydrogens, the molecule conserves double bonds.
        """
        submolecule = Chem.MolFromSmarts("[#6v4]#[#7v3]")
        matches = self.molecule.GetSubstructMatches(submolecule, maxMatches=5)

        for match in matches:
            bond = self.molecule.GetBondBetweenAtoms(match[0], match[1])
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
            
        Chem.SanitizeMol(self.molecule)

        for match in matches:
            bond = self.molecule.GetBondBetweenAtoms(match[0], match[1])
            bond.SetBondType(Chem.rdchem.BondType.TRIPLE)
           
            self.molecule.GetAtomWithIdx(match[0]).SetNumRadicalElectrons(0)
            self.molecule.GetAtomWithIdx(match[1]).SetNumRadicalElectrons(0)
            
            if self.molecule.GetAtomWithIdx(match[0]).HasProp("_displayLabel"):
                self.molecule.GetAtomWithIdx(match[0]).SetProp("_displayLabel", "C")
            self.molecule.GetAtomWithIdx(match[0]).SetIntProp("totalNumHs", 0)

            self.molecule.GetAtomWithIdx(match[1]).SetProp("_displayLabel", "N")
            self.molecule.GetAtomWithIdx(match[1]).SetIntProp("totalNumHs", 0)
        
        rdDepictor.Compute2DCoords(self.molecule)
     
    def add_explicit_carbons_with_triple_bond(self):
        submolecule = Chem.MolFromSmarts("[#6v4]#[#7v3]")
        matches = self.molecule.GetSubstructMatches(submolecule, maxMatches=5)

        for match in matches:
            atom = self.molecule.GetAtomWithIdx(match[0])
            atom.SetProp("_displayLabel", "C")

        submolecule = Chem.MolFromSmarts("[#6v4]#[#6v4]")
        matches = self.molecule.GetSubstructMatches(submolecule, maxMatches=5)

        for match in matches:
            atom_C1 = self.molecule.GetAtomWithIdx(match[0])
            atom_C2 = self.molecule.GetAtomWithIdx(match[1])
            atom_C1.SetProp("_displayLabel", "C")     
            atom_C2.SetProp("_displayLabel", "C")
            
    def add_explicit_carbons(self):
        explicit_carbons_proportion = 0.25
        for atom in self.molecule.GetAtoms():
            if (atom.GetFormalCharge() == 0) and (atom.GetSymbol() == "C") and coin_flip(explicit_carbons_proportion):
                number_hydrogens = atom.GetTotalNumHs()
                if number_hydrogens == 0:
                    h_string = ""
                if number_hydrogens == 1:
                    h_string = "H"
                if number_hydrogens > 1:
                    h_string = "H<sub>" + str(number_hydrogens) + "</sub>"

                if atom.HasProp("totalNumHs"):
                    # Do not overwrite modifications from angle_carbons_with_triple_bond
                    atom.SetProp("_displayLabel", "C")  
                else:
                    atom.SetProp("_displayLabel", "C" + h_string)    
                
    def add_all_explicit_carbons(self):
        for atom in self.molecule.GetAtoms():
            if (atom.GetFormalCharge() == 0) and (atom.GetSymbol() == "C"):
                number_hydrogens = atom.GetTotalNumHs()
                if number_hydrogens == 0:
                    h_string = ""
                if number_hydrogens == 1:
                    h_string = "H"
                if number_hydrogens > 1:
                    h_string = "H<sub>" + str(number_hydrogens) + "</sub>"

                if atom.HasProp("totalNumHs"):
                    # Do not overwrite modifications from angle_carbons_with_triple_bond
                    atom.SetProp("_displayLabel", "C")  
                else:
                    atom.SetProp("_displayLabel", "C" + h_string)     
  
    def transform(self):
        """
        The input molecule is modified randomly by : 
             - selecting a random conformation
             - adding bonds between N, O, S and H
             - reducing the size of bonds with explicit H
             - removing H from abreviations like "OH" or "NH2"
        Conformers are generated only for molecule with low complexity.
        If the size of bonds with explicit Hs is reduced, the bond size, the font size and the atom padding are set to low values.
        """    
        small_molecules_proportion = 0.2
        conformer_proportion = 0.5
        explicit_h_proportion = 0.25
        reduce_h_bonds_size_proportion = 0.2
        remove_h_from_abbreviations_proportion = 0.05
        small_circled_charges_proportion = 0.05
        large_circled_charges_proportion = 0.15
        left_charges_proportion = 0.1
        if self.fake_molecule:
            abbreviations_multiple_attachments_extension_proportion = 0.1
            abbreviations_single_attachment_extension_proportion = 0.3      # Best 0.25
        else:
            abbreviations_multiple_attachments_extension_proportion = 0
            abbreviations_single_attachment_extension_proportion = 0
        explicit_carbons_with_triple_bond_proportion = 0.25
        angle_carbons_with_triple_bond_proportion = 0.15
        explicit_carbons_proportion = 0.20
        all_explicit_carbons_proportion = 0.03
        explicit_h_c_proportion = 0.20
        
        if (self.molecule_size <= 7) and coin_flip(conformer_proportion):
            #print("a", conformer_proportion)
            logger.debug("Random conformer selection")   
            self.select_random_conformer()
            # Other transformations are not performed on conformers
            return self.molecule

        probability_a = explicit_h_proportion/(1 - conformer_proportion*small_molecules_proportion)
        explicit_h = False
        if coin_flip(probability_a):
            #print("b", probability_a)
            logger.debug("Add bonds between P, B, N, O, S and H") 
            self.add_explicit_h()
            explicit_h = True
            
        elif coin_flip(remove_h_from_abbreviations_proportion/((1 - probability_a)*(1 - conformer_proportion*small_molecules_proportion))):
            #print("d", remove_h_from_abbreviations_proportion/((1 - probability_a)*(1 - conformer_proportion*small_molecules_proportion)))
            logger.debug("Remove H from abreviations") 
            self.remove_h_from_abbreviations()
        
        if coin_flip(explicit_h_c_proportion/(1 - conformer_proportion*small_molecules_proportion)):
            #print("b", probability_a)
            logger.debug("Add bonds between C and H") 
            self.add_explicit_h_c()
            explicit_h = True
            
        probability_b = (small_circled_charges_proportion + large_circled_charges_proportion)/(1 - conformer_proportion*small_molecules_proportion)
        circled_charges_added = False
        if coin_flip(probability_b):
            #print("e", probability_b)
            logger.debug("Add circled charges") 
            if coin_flip(large_circled_charges_proportion/(small_circled_charges_proportion + large_circled_charges_proportion)):
                charge_size = "large"
            else:
                charge_size = "small"
            self.add_circled_charges(charge_size)
            circled_charges_added = True

        elif coin_flip(left_charges_proportion/((1 - probability_b)*(1 - conformer_proportion*small_molecules_proportion))):
            #print("f", left_charges_proportion/((1 - probability_b)*(1 - conformer_proportion*small_molecules_proportion)))
            logger.debug("Move charges to the left side") 
            self.move_charges_to_left_side()

        if coin_flip((abbreviations_multiple_attachments_extension_proportion + abbreviations_single_attachment_extension_proportion)/(1 - conformer_proportion*small_molecules_proportion)):
            #print("g", (abbreviations_multiple_attachments_extension_proportion + abbreviations_single_attachment_extension_proportion)/(1 - conformer_proportion*small_molecules_proportion))
            logger.debug("Add abbreviations extensions") 
            if coin_flip(abbreviations_multiple_attachments_extension_proportion/(abbreviations_multiple_attachments_extension_proportion + abbreviations_single_attachment_extension_proportion)):
                abbreviation_size = "multiple_attachments"
            else:
                abbreviation_size = "single_attachment"
            self.add_abbreviations_extension(abbreviation_size, circled_charges_added)
        
        if coin_flip(explicit_carbons_proportion/(1 - conformer_proportion*small_molecules_proportion)):
            logger.debug("Add explicit carbons") 
            self.add_explicit_carbons()
        
        if coin_flip(explicit_carbons_with_triple_bond_proportion/(1 - conformer_proportion*small_molecules_proportion)):
            #print("h", explicit_carbons_with_triple_bond_proportion/(1 - conformer_proportion*small_molecules_proportion))
            logger.debug("Add explicit carbons when connected to a nitrogen with a triple bond") 
            self.add_explicit_carbons_with_triple_bond()
            
        if coin_flip(angle_carbons_with_triple_bond_proportion/(1 - conformer_proportion*small_molecules_proportion)):
            #print("h", explicit_carbons_with_triple_bond_proportion/(1 - conformer_proportion*small_molecules_proportion))
            logger.debug("Angle the triple bond between carbons and nitrogens") 
            self.angle_carbons_with_triple_bond()
        
        if coin_flip(all_explicit_carbons_proportion/(1 - conformer_proportion*small_molecules_proportion)):
            logger.debug("Add all explicit carbons") 
            self.add_all_explicit_carbons()
            
        if explicit_h and coin_flip(reduce_h_bonds_size_proportion/explicit_h_proportion):
            # The reduction of h bond size has to be done after the abbreviations extension, which call rdDepictor.Compute2DCoords(molecule) 
            #print("c", reduce_h_bonds_size_proportion/explicit_h_proportion)
            logger.debug("Reduce size of bonds between P, B, N, O, S and H")
            self.reduce_h_bonds_size()
                
        return self.molecule