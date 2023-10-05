#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import numpy as np

# RDKit
from rdkit import Chem
from rdkit.Chem import rdDepictor

def is_aromatic(mol, bondRing):
    """
    Assess if a ring is aromatic or not.
    """
    for id_ in bondRing:
        if (not mol.GetBondWithIdx(id_).GetIsAromatic()) or (str(mol.GetBondWithIdx(id_).GetBondType()) == "TRIPLE"):
            return False
    return True
    
def get_molecule_size(molecule): 
    """
    Computes the size of a molecule using its min-max coordinates.
    """
    if molecule.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(molecule)
  
    for conformer in molecule.GetConformers():
        coordinates = conformer.GetPositions()
        if len(coordinates) < 1:
            return 0
        size = np.amax(coordinates, axis=0) - np.amin(coordinates, axis=0)
        return max(size)

def get_molecule_from_smiles(smiles, remove_stereochemistry):
    """
    Convert SMILES to MDL Molfile, which is a format holding coordinates of atoms and bonds.
    Read SMILES by removing stereochemistry and resolving aromaticity.

    Chem.MolFromSmiles()
          - clean up, setConjugation, setHybridization, cleanupChirality, adjustHs
          - kekulize (converts aromatic rings to their Kekule form)
          - updatePropertyCache (calculates the explicit and implicit valences on all atoms)
          - assignRadicals determines the number of radical electrons
          - setAromaticity (sets the aromatic flag on atoms and bonds)
          - calculates the hybridization state of each atom

    Some more details:
          - clearComputedProps: removes any computed properties that already exist
            on the molecule and its atoms and bonds. This step is always performed.

          - cleanUp: standardizes a small number of non-standard valence states.

          - updatePropertyCache: calculates the explicit and implicit valences on all atoms. This generates exceptions for atoms in higher-than-allowed valence states. This step is always performed, but if it is “skipped” the test for non-standard valences will not be carried out.

          - symmetrizeSSSR: calls the symmetrized smallest set of smallest rings algorithm (discussed in the Getting Started document).

          - Kekulize: converts aromatic rings to their Kekule form. Will raise an exception if a ring cannot be kekulized or if aromatic bonds are found outside of rings.

          - assignRadicals: determines the number of radical electrons (if any) on each atom.

          - setAromaticity: identifies the aromatic rings and ring systems (see above), sets the aromatic flag on atoms and bonds, sets bond orders to aromatic.
          - setConjugation: identifies which bonds are conjugated

          - setHybridization: calculates the hybridization state of each atom

          - cleanupChirality: removes chiral tags from atoms that are not sp3 hybridized.

          - adjustHs: adds explicit Hs where necessary to preserve the chemistry. This is typically needed for heteroatoms in aromatic rings. The classic example is the nitrogen atom in pyrrole.
            
    Chem.Kekulize(molecule, clearAromaticFlags=True)
          - kekulize is used to clear aromatic flags
    """  

    # sanitize = False is mandatory for H abreviations
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    
    if molecule is None:
        return None

    if remove_stereochemistry:
        Chem.RemoveStereochemistry(molecule)
        
    molecule.UpdatePropertyCache(strict=False)
    sanity = Chem.SanitizeMol(molecule, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
                                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
                                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)

    if sanity != Chem.rdmolops.SANITIZE_NONE:
        return None

    return molecule


def get_molecule_from_molfile(molfile, remove_stereochemistry):
    """
    Read MDL Molfile
    """
    molecule = Chem.MolFromMolFile(molfile, sanitize=False)
    if molecule is None:
        return None
    
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(molecule)
        
    molecule.UpdatePropertyCache(strict=False)
    sanity = Chem.SanitizeMol(molecule, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
                                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
                                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
    
    if sanity != Chem.rdmolops.SANITIZE_NONE:
        return None
    
    return molecule

def compute_nb_heavy_atoms_and_explicit_hs(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    molecule.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(molecule,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                     Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|\
                     Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
                     Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)

    nb_heavy_atoms_and_explicit_hs = 0
    for i, atom in enumerate(molecule.GetAtoms()):
        if (atom.GetSymbol() == "H" or atom.GetNumExplicitHs() > 0):
            nb_heavy_atoms_and_explicit_hs += 1
        elif (atom.GetSymbol() != "H"):
            nb_heavy_atoms_and_explicit_hs += 1   

    return nb_heavy_atoms_and_explicit_hs