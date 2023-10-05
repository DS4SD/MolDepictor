#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RDKit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Geometry import Point3D

# Maths
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Python Standard Library
import re
from itertools import combinations
import copy
from collections import defaultdict

# Others
from shapely.geometry import LineString

RDLogger.DisableLog('rdApp.*')


class LabelMolFile:
    def __init__(self, filename, reduce_abbreviations=True) -> None:
        """
        reduce_abbreviations is indeed mandatory, need some refactoring.
        """
        self.fragments = []
        self.label_molfile_fragments = []
        
        self.rdk_mol = Chem.MolFromMolFile(str(filename), sanitize=False, removeHs=False)
        if self.rdk_mol == None:
            print("The molecule can't be read with RDKt")
            return 

        self.mol = self._read_molfile(filename)
        
        if self.mol and reduce_abbreviations:
            #molecule = Chem.MolFromMolFile(filename)
            #self.rdk_mol = rdAbbreviations.CondenseAbbreviationSubstanceGroups(molecule)
            
            self.reduce_abbreviations()
            fragments = Chem.GetMolFrags(self.rdk_mol)
            self.fragments = self.join_intersecting_fragments(fragments)
            fragments = self.unreduce_fragments(self.fragments)
            # Fragments molfile annotators doesn't have condensed abbreviations
            self.set_label_molfile_fragments(fragments)
            #pprint(self.mol)
            #for indices, frags in zip(fragments, self.label_molfile_fragments):
            #    if frags[1]:
            #        draw_molecule(frags[1], path=str(indices) + ".png", display_indices=True)
            #print(self.fragments)
            #draw_molecule(self.rdk_mol, path="rdk_mol.png", display_indices=True)
        return

    def unreduce_fragments(self, fragments):
        unreduced_fragments = []
        offset = 0
        for fragment in fragments:
            unreduced_fragment = []
            for index in fragment:
                unreduced_fragment.append(index + offset)
                atom = self.rdk_mol.GetAtomWithIdx(index)
                try:
                    indices = [int(i) for i in atom.GetProp("condensed_indices").split(",")]
                    unreduced_fragment.extend(indices)
                    offset += len(indices)
                except:
                    pass
            unreduced_fragments.append(unreduced_fragment)
        return unreduced_fragments
                    
    def get_info(self):
        are_markush = self.are_markush() 
        are_molecules = self.are_molecules()
        are_abbreviated = self.are_abbreviated()
        rdk_mols = self.get_rdk_mols()
        return [
            {
                "is_molecule": are_molecules[index], 
                "is_markush": are_markush[index], 
                "is_abbreviated": are_abbreviated[index],
                "rdk_mol": rdk_mols[index],
            } for index in range(len(are_molecules))
        ]

    def how_many_structures(self):
        if self.rdk_mol is None:
            return -1
        else:
            fragments = Chem.GetMolFrags(self.rdk_mol)
            return len(fragments)

    def are_molecules(self):
        if not self.label_molfile_fragments:
            return [False]
        are_molecules = [label_molfile_fragment.is_molecule() for label_molfile_fragment, _ in self.label_molfile_fragments]
        return are_molecules

    def are_markush(self):
        if not self.label_molfile_fragments:
            return [False]
        are_markush = [label_molfile_fragment.is_markush() for label_molfile_fragment, _ in self.label_molfile_fragments]
        return are_markush

    def are_abbreviated(self):
        if not self.label_molfile_fragments:
            return [False]
        are_abbreviated = [label_molfile_fragment.is_abbreviated() for label_molfile_fragment, _ in self.label_molfile_fragments]  
        return are_abbreviated

    def get_rdk_mols(self):
        if not self.label_molfile_fragments:
            return [None]
        return [rdk_mol for _, rdk_mol in self.label_molfile_fragments]  

    def set_label_molfile_fragments(self, fragments):
        for fragment in fragments:
            # Filter non continuous indices in fragment 
            if not sorted(fragment) == list(range(min(fragment), max(fragment)+1)):
                print(f"The fragment {fragment} contains non consecutive indices")
                self.label_molfile_fragments.append(
                    (LabelMolFileFragment(rdk_mol=None, mol=None), None,)
                )
                continue
            # Crop RDKit molecule
            molecule_editable = Chem.EditableMol(self.unreduced_rdk_mol)
            for index in range(self.unreduced_rdk_mol.GetNumAtoms()-1, -1, -1):
                if index not in fragment:
                    molecule_editable.RemoveAtom(index)
            rdk_mol_fragment = molecule_editable.GetMol()
            # Crop mol descriptor
            mol_fragment = self.crop_mol_descriptor(fragment)
            self.label_molfile_fragments.append(
                (LabelMolFileFragment(rdk_mol=rdk_mol_fragment, mol=mol_fragment), rdk_mol_fragment,)
            )
        return 

    def crop_mol_descriptor(self, fragment):
        mol_fragment = {'conn': [], 'coord': [], 'sgroups': [], 'a_sections': [], 'chg': [], 'als': [], 'apo': [], 'rad': [], 'iso': []}
        # Atoms
        for entry in self.mol['coord']:
            if entry['id'] in fragment:
                new_entry = entry.copy()
                new_entry['id'] = entry["id"] - fragment[0]
                mol_fragment['coord'].append(new_entry)
        mol_fragment['natm'] = len(mol_fragment['coord'])
        # Bonds
        for entry in self.mol['conn']:
            if (entry['atm1'] in fragment) and (entry['atm2'] in fragment):
                new_entry = entry.copy()
                new_entry['atm1'] = entry['atm1'] - fragment[0]
                new_entry['atm2'] = entry['atm2'] - fragment[0]
                mol_fragment['conn'].append(new_entry)
        mol_fragment['nbnd'] = len(mol_fragment['conn'])
        # A groups
        for a_section in self.mol['a_sections']:
            if a_section['a_atom'] in fragment:
                new_a_section = a_section.copy()
                new_a_section["a_atom"] = a_section["a_atom"] - fragment[0] # Possibility to use a mapping
                mol_fragment['a_sections'].append(new_a_section)
        # S Groups
        for sgroup in self.mol['sgroups']:
            if all(atom_index in fragment for atom_index in sgroup["sal_atoms_list"]):
                new_sgroup = sgroup.copy()
                new_sgroup["sal_atoms_list"] = [atom_index - fragment[0] for atom_index in sgroup["sal_atoms_list"]] # Possibility to use a mapping
                mol_fragment['sgroups'].append(new_sgroup)
        # Attachment points
        for attachment_point in self.mol['apo']:
            if attachment_point in fragment:
                mol_fragment['apo'].append(attachment_point)
        # Charges
        for charge in self.mol['chg']:
            if charge in fragment:
                mol_fragment['chg'].append(charge)
        # Radicals
        for radical in self.mol['rad']:
            if radical in fragment:
                mol_fragment['rad'].append(radical)
        # Isotopes
        for isotope in self.mol['iso']:
            if isotope in fragment:
                mol_fragment['iso'].append(isotope)
        return mol_fragment

    @staticmethod
    def _read_molfile(filename):
        """
        defintion of MOL file: 
           https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md
        Properties block:
            M ALS - atom list and exclusive list
            M APO - Rgroup attachment point
            M CHG - charge
            M RAD - radical
            M ISO - isotope mass numbers
            M RGP - Rgroup labels on root structure
            M LOG - Rgroup logic
            M LIN - link nodes
            M SUB - substitution count query property (s)
            M UNS - unsaturated atom query property (u)
            M RBC - ring bond count query property (rb)
            M STY - Sgroup type
            M SST - Sgroup subtype
            M SCN - Sgroup connectivity (head-to-head, head-to-tail or either/unknown)
            M SAL - atoms that define the Sgroup
            M SPA - multiple group parent atom list (paradigmatic repeating unit atoms)
            M SBL - Sgroup's crossing bonds
            M SMT - Sgroup label
            M SPL - Sgroup parent list
            M SDS EXP - Sgroup expansion
            M SDT - Data sgroup field description
            M SDD - Data sgroup display information
            M SCD - Data sgroup data
            M SED - Data sgroup data end of line
            M SNC - Sgroup component numbers
            M CRS - Sgroup correspondence
            M SDI - display coordinates in each S-group bracket
            M SBT - the displayed S-group bracket style
            M SAP - the S-group attachment point information
            M MRV SMA - SMARTS H, X, R, r, a, A properties (Marvin extension)
            A- Atom alias
            V- Atom value
        00113004.cdx
        ChemDraw12302019272D
        200236  0  0  0  0  0  0  0  0999 V2000
        -0.2171   11.6302    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        -0.2171   11.0370    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            0.2966   10.7404    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        [...]
        [...]
        0         1         2         3         4         5         6
        0123456789012345678901234567890123456789012345678901234567890123456789
            1.7176   -1.9616    0.0000 R26 0  0  0  0  0  0  0  0  0  0  0  0
        [...]
        00023001.cdx
        ChemDraw01062116192D
        16 13  0  0  0  0  0  0  0  0999 V2000
        -2.1562   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            2.1170   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        -2.6982   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        -1.8732   -0.4125    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0
        -1.8732    0.4125    0.0000 R1  0  0  0  0  0  0  0  0  0  0  0  0
        -1.8732   -1.2375    0.0000 R2  0  0  0  0  0  0  0  0  0  0  0  0
        -0.8994   -0.4125    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
        -0.0744   -0.4125    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0
        -0.0744    0.4125    0.0000 R5  0  0  0  0  0  0  0  0  0  0  0  0
        -0.0744   -1.2375    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
            0.8994   -0.4125    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
        -0.0744    1.2375    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
            1.7244   -0.4125    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0
            1.7244    0.4125    0.0000 R3  0  0  0  0  0  0  0  0  0  0  0  0
            1.7244   -1.2375    0.0000 R4  0  0  0  0  0  0  0  0  0  0  0  0
            2.6982   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        3  4  1  0      
        4  5  1  0      
        4  6  1  0      
        4  7  1  0      
        7  8  1  0      
        8  9  1  0      
        8 10  1  0      
        8 11  1  0      
        9 12  1  0      
        11 13  1  0      
        13 14  1  0      
        13 15  1  0      
        13 16  1  0      
        A    1
        [
        A    2
        ]
        M  END
        """
        try:
            mol = {}
            with open(filename, 'rb') as fid:
                _ = fid.readline()
                _ = fid.readline()
                _ = fid.readline()
                
                l = fid.readline()
                
                mol['natm'], mol['nbnd'] = int(l[0:3]), int(l[3:6])

                atom_index = 0
                coord = []
                for _ in range(mol['natm']):
                    l = fid.readline()
                    x, y, z = float(l[0:10]), float(l[10:20]), float(l[20:30])
                    atm = l[30:34].decode("utf-8","replace").strip()
                    coord.append({'atm': atm, 'xyz': np.array([x,y,z]), 'id': atom_index})
                    atom_index += 1
                mol['coord'] = coord

                conn = []
                for _ in range(mol['nbnd']):
                    l = fid.readline()
                    a1, a2, bt, bs = int(l[0:3]) - 1, int(l[3:6]) - 1, int(l[6:9]), int(l[9:12])
                    if a1 >= mol['natm']:
                        assert f'wrong atom number {a1}, total atoms {mol["natm"]}'
                    if a2 >= mol['natm']:
                        assert f'wrong atom number {a2}, total atoms {mol["natm"]}'
                    conn.append({'atm1': a1, 'atm2': a2, 'bt': bt, 'bs': bs})
            
                mol['conn'] = conn
                a_sections = []
                sgroups = {}
                charges, attachment_points, atoms_list, radicals, isotopes = [], [], [], [], []

                file_lines = fid.readlines()
                lines = []
                found_end = False
                for line in file_lines:
                    line = line.decode("utf-8","replace").strip()
                    lines.append(line)
                    if 'M  END' in line:
                        found_end = True
                        break

                if not found_end:
                    return None

                a_section_index = 0
                for index in range(len(lines)):
                    line = lines[index]
                    line_list = [character for character in line.split(" ") if character != '']

                    if line_list[0] == "A":
                        next_line = lines[index + 1]
                        a_sections.append({"a_section_index": a_section_index,
                                           "a_atom": int(line_list[1]) - 1,
                                           "a_label": next_line})
                        index += 1
                        a_section_index += 1
                        continue
                    
                    if (line_list[0] == "M") and (len(line_list) > 2) and (re.search(r'\b[A-Z]{3}\b', line_list[1])):
                        # S group
                        if line_list[1] == "STY":
                            # Create new s group
                            sgroups[line_list[3]] = {}
                            sgroups[line_list[3]]["sdi"] = []
                            sgroups[line_list[3]]["type"] = line_list[-1].lower() # GEN, SUP, SRU, ...
                            sgroups[line_list[3]]["sal_atoms_list"] = []
                            sgroups[line_list[3]]["slb_bonds_list"] = []
                            sgroups[line_list[3]]["smt_label"] = ""

                        if line_list[1] == "SAL":
                            if len(sgroups[line_list[2]]["sal_atoms_list"]) == 0:
                                sgroups[line_list[2]]["sal_atoms_list"] = [int(id) - 1 for id in line_list[4:]]
                            else:
                                sgroups[line_list[2]]["sal_atoms_list"].extend([int(id) - 1 for id in line_list[4:]])
                        
                        if line_list[1] == "SBL":
                            sgroups[line_list[2]]["slb_bonds_list"] = [int(id) - 1 for id in line_list[4:]]

                        if line_list[1] == "SMT":
                            sgroups[line_list[2]]["smt_label"] = line_list[3]
                        
                        if line_list[1] == 'CHG':
                            charges.append(int(line_list[3]) - 1)

                        if line_list[1] == 'RAD':
                            radicals.append(int(line_list[3]) - 1)

                        if line_list[1] == "APO":
                            attachment_points.append(int(line_list[3]) - 1)

                        if line_list[1] == "ISO":
                            isotopes.append(int(line_list[3]) - 1)

                        if line_list[1] == "SDI":
                            sgroups[line_list[2]]["sdi"].append(1) # Filler

                        if line_list[1] == "ALS":
                            atoms_list.append(1) # Filler 

                sgroups_list = []
                for key in sgroups:
                    sgroups_list.append({"sgroup_index": key,
                                         "sal_atoms_list": sgroups[key]["sal_atoms_list"],
                                         "smt_label": sgroups[key]["smt_label"],
                                         "sdi": sgroups[key]["sdi"],
                                         "sty_type": sgroups[key]["type"],
                                         })
                mol['chg'] = charges
                mol['als'] = atoms_list
                mol['apo'] = attachment_points
                mol['rad'] = radicals
                mol['iso'] = isotopes
                mol['sgroups'] = sgroups_list
                mol['a_sections'] = a_sections
        except Exception as e:
            mol = None
            print(f"Can't parse Molfile manually: {str(e)}")
        return mol
    
    def join_intersecting_fragments(self, fragments):
        """
        Join a multicenter S-group of length less than 3 with its parent molecule.
        Notes: 
            Fragments indices have to be sorted
        """
        fragments_connections = defaultdict(list)

        for fragment_index, fragment in enumerate(fragments):
            connected = False
            if len(fragment) < 4:
                for index_1, index_2 in combinations(fragment, 2):
                    if self.rdk_mol.GetBondBetweenAtoms(index_1, index_2) != None:
                        atom_1_coordinates = self.rdk_mol.GetConformer(0).GetAtomPosition(index_1)
                        atom_2_coordinates = self.rdk_mol.GetConformer(0).GetAtomPosition(index_2)
                        line = LineString([atom_1_coordinates, atom_2_coordinates])
                        for super_fragment_index, super_fragment in enumerate(fragments):
                            if super_fragment != fragment:
                                for super_index_1, super_index_2 in combinations(super_fragment, 2):
                                    if self.rdk_mol.GetBondBetweenAtoms(super_index_1, super_index_2) != None:
                                        bond = self.rdk_mol.GetBondBetweenAtoms(super_index_1, super_index_2)
                                        if ((bond.GetBeginAtom().GetIdx() != index_1) and (bond.GetEndAtom().GetIdx() != index_2)) and \
                                            ((bond.GetBeginAtom().GetIdx() != index_2) and (bond.GetEndAtom().GetIdx() != index_1)):
                            
                                            atom_1_coordinates_query = self.rdk_mol.GetConformer(0).GetAtomPosition(bond.GetBeginAtom().GetIdx())
                                            atom_2_coordinates_query = self.rdk_mol.GetConformer(0).GetAtomPosition(bond.GetEndAtom().GetIdx())
                                            line_query = LineString([atom_1_coordinates_query, atom_2_coordinates_query])
                                            if line.intersects(line_query):
                                                #print(f"{index_1}, {index_2} intersects with {super_index_1}, {super_index_2}")
                                                fragments_connections[super_fragment_index].append(fragment_index)
                                                connected = True
                                                break
                            if connected:
                                break
                    if connected:
                        break
            if connected == False and fragment_index not in fragments_connections:
                fragments_connections[fragment_index] = []

        fragments_extended = []
        for super_fragment_index, sub_fragment_indices in fragments_connections.items():
            fragment_extended = list(fragments[super_fragment_index])
            for sub_fragment_index in sub_fragment_indices:
                for index in fragments[sub_fragment_index]:
                    fragment_extended.append(index)
            fragments_extended.append(tuple(sorted(fragment_extended)))
        return tuple(fragments_extended)

    def reduce_abbreviations(self):
        self.unreduced_rdk_mol = copy.deepcopy(self.rdk_mol)
        connection_atoms_abbreviations = []
        for sgroup in self.mol["sgroups"]:
            if sgroup["sdi"] == False:
                connection_atoms_abbreviations.append((sgroup["sal_atoms_list"], sgroup["smt_label"]))
        for a_section in self.mol["a_sections"]:
            connection_atoms_abbreviations.append(([a_section["a_atom"]], a_section["a_label"]))
        
        # To avoid shifting indices during the molecule modifications, changes are made starting by larger indices
        connection_atoms_abbreviations = sorted(connection_atoms_abbreviations, key = lambda x: x[0][0], reverse=True)

        for connection_atoms, abbreviation in connection_atoms_abbreviations:
            # Remove abbreviated atoms and connect the "abbreviation-atom" to its attachment points
            if len(connection_atoms) > 1:
                #print("The molecule contains a superatom with multiple attachment points.")
                attachments = []
                for connection_atom in connection_atoms:
                    for bond in self.rdk_mol.GetAtomWithIdx(connection_atom).GetBonds():
                        for atom in [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]:
                            if atom not in connection_atoms:
                                attachments.append({"index": atom, "type": bond.GetBondType()})

                if len(attachments) > 1:
                    conformer = self.rdk_mol.GetConformer()
                    attachment_positions = []
                    for attachment in (attachments):
                        position = conformer.GetAtomPosition(attachment["index"])
                        attachment_positions.append((position.x, position.y, position.z))
                    average_position = [np.mean([x for x, y, z in attachment_positions]), \
                                        np.mean([y for x, y, z in attachment_positions]), \
                                        np.mean([z for x, y, z in attachment_positions])]
                    conformer.SetAtomPosition(connection_atoms[0], Point3D(*average_position))
                
                molecule_editable = Chem.EditableMol(self.rdk_mol)

                for attachment in attachments:
                    if not self.rdk_mol.GetBondBetweenAtoms(connection_atoms[0], attachment["index"]):
                        molecule_editable.AddBond(connection_atoms[0], attachment["index"], order=attachment["type"])

                remove_indices = connection_atoms[1:]
                for index in range(self.rdk_mol.GetNumAtoms()-1, -1, -1):
                    if index in remove_indices:
                        molecule_editable.RemoveAtom(index)

                self.rdk_mol = molecule_editable.GetMol()   
                self.rdk_mol.GetAtomWithIdx(connection_atoms[0]).SetProp("condensed_indices", ','.join([str(index) for index in connection_atoms[1:]]))
            
            # Add abbreviations
            abbreviation = re.sub(r'(?<!<)(?:\\S)(.*?)(\\n|\\S|\<s|$)', r'<sup>\1</sup>\2', abbreviation)                                                                                                          
            abbreviation = re.sub(r'(?<!<)(?:\\s)(.*?)(\\n|\\S|\<s|$)', r'<sub>\1</sub>\2', abbreviation)
            abbreviation = re.sub(r'(\\n)', r'', abbreviation)
            abbreviation = abbreviation.replace("\r", "\n")
            if abbreviation[0:3] == "<su":
                abbreviation = " " + abbreviation
            self.rdk_mol.GetAtomWithIdx(connection_atoms[0]).SetProp("_displayLabel", abbreviation)
            self.rdk_mol.GetAtomWithIdx(connection_atoms[0]).SetProp("_displayLabelW", abbreviation)
           
        return 

class LabelMolFileFragment:

    # Number of atoms in a chain
    min_mark_chain = 6

    markush_bond_types = [1, 4, 6]

    symbols = ['H','D','He','Li','Be','B','C','N','O','F','Ne',
               'Na','Mg','Al','Si','P','S','Cl','Ar_removed_','K','Ca',
               'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
               'Ga','Ge','As','Se','Br','Kr','Rb_removed_','Sr','Y_removed_','Zr',
               'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
               'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
               'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
               'Lu','Hf','Ta','W_remove_','Re','Os','Ir','Pt','Au','Hg',
               'Tl','Pb','Bi','Po','At','Rn_removed_','Fr','Ra_removed_','Ac','Th',
               'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',
               'Md','No','Lr',]

    # For all labels
    # all_irrelevant_extra = ['CHG', 'RAD']

    # Skip Re, Rh, Ru
    # R1a
    markush_symbols = ['\*', 'Ar\d{0,2}', 'R\d.{0,2}', 'R[a-dfgi-tv-z]?\d{0,2}[,\.-;²¢]?\d{0,2}', '[ONS]R\d{0,2}', '[ALQRWXZ]\d{0,2}']

    """
    max_subscript = 100
    markush_symbols = ['*', 'Rn'] + \
        ['Ar']+[f'Ar{str(i)}' for i in range(max_subscript)] + \
        ['Rb']+[f'Rb{str(i)}' for i in range(max_subscript)] + [f'R{str(i)}b' for i in range(max_subscript)] + \
        ['Ra']+[f'Ra{str(i)}' for i in range(max_subscript)] + [f'R{str(i)}a' for i in range(max_subscript)] + \
        ['Rf']+[f'Rf{str(i)}' for i in range(max_subscript)] + [f'R{str(i)}f' for i in range(max_subscript)] + \
        ['A']+[f'A{str(i)}' for i in range(max_subscript)] + \
        ['R']+[f'R{str(i)}' for i in range(max_subscript)] + \
        ['Q']+[f'Q{str(i)}' for i in range(max_subscript)] + \
        ['X']+[f'X{str(i)}' for i in range(max_subscript)] + \
        ['W']+[f'W{str(i)}' for i in range(max_subscript)] + \
        ['Z']+[f'Z{str(i)}' for i in range(max_subscript)]
    """

    _markush_smt_symbols_base_lower = ['b', 'c', 'd', 'l', 'm', 'n', 'o', 'p', 'r', 't', 'u', 'w', 'x', 'y', 'z']
    _markush_smt_symbols_base = _markush_smt_symbols_base_lower + [s.upper() for s in _markush_smt_symbols_base_lower if s not in ['n', 't', 'w', 'u']]
    markush_smt_symbols = \
        _markush_smt_symbols_base_lower + \
        [s+'\d{0,2}' for s in _markush_smt_symbols_base] + \
        [s+'\+[1-5]' for s in _markush_smt_symbols_base] + \
        [s+'\-[1-5]' for s in _markush_smt_symbols_base] + \
        ['\d{1,3}'] + \
        ['\d{1,3}\.\d{1,2}'] + \
        ['\d{1,2}-\d{1,2}'] + \
        ['OR\d{0,2}'] + \
        ['[^{1,2}]\s{1,2}'] 
        

    markush_relevant_extra = ['APO']

    all_force_kill_extra = ['ALS']
    
    abbreviation_smt_symbols = []

    def __init__(self, rdk_mol, mol):
        self.rdk_mol = rdk_mol
        self.mol = mol
        return

    def _has_more_than_one_atom(self):
        return len(self.mol['coord']) > 1

    def _has_only_symbols(self):
        for coord in self.mol['coord']:
            if coord['atm'] not in self.symbols:
                return False
        return True

    def _has_brackets(self):
        for sgroup in self.mol['sgroups']:
            if len(sgroup["sdi"]) != 0:
                return True
        return False
    
    def _has_markush_symbols(self):
        for coord in self.mol['coord']:
            if re.search('^('+"|".join(self.markush_symbols)+')(\s|$)', coord['atm']):
                return True
        return False

    def _has_markush_relevant_extra(self):
        # Relevant fields: 'APO', 'STY'+'GEN'
        if len(self.mol["apo"]) > 0 or \
           ( len(self.mol["sgroups"]) > 0 and all([sg['sty_type'] == 'gen' for sg in self.mol['sgroups']]) ):
            return True
        return False

    def _has_only_irrelevant_extra(self):
        # Irrelevant fields: ['CHG', 'RAD']
        if (len(self.mol["sgroups"]) == 0) and (len(self.mol["a_sections"]) == 0) and (len(self.mol["iso"]) == 0) and \
            (len(self.mol["als"]) == 0) and (len(self.mol["apo"]) == 0):
            return True
        return False
    
    def _has_force_kill_extra(self):
        if len(self.mol["als"]) > 0:
            return True
        return False

    def _has_markush_smt_symbols(self):
        """
        from US10881743-20210105-C00006.MOL
        [...]
        M  SMT   1 n
        [...]
        BUT NOT
        from US20210017156A1-20210121-C00032.MOL
        [...]
        M  SMT   1 Boc
        [...]
        if re.search('^\s*M\s+SMT\s+\d{1,4}\s+('+"|".join(self.markush_smt_symbols)+')(\s|$)', line):
        """
        for sgroup in self.mol["sgroups"]:
            test_label = sgroup["smt_label"].replace("\r","")
            if re.search('^('+"|".join(self.markush_smt_symbols)+')(\s|$)', test_label):
                return True
        for a_section in self.mol["a_sections"]:
            test_label = a_section["a_label"].replace("\r","")
            if re.search('^('+"|".join(self.markush_smt_symbols)+')(\s|$)', test_label):
                return True
        return False
    
    def is_molecule(self):
        if self.rdk_mol is None:
            return False

        if self._has_force_kill_extra():
            return False

        all_sym = self._has_only_symbols()
        has_more_than_one_atom = True #self._has_more_than_one_atom()
        has_irrelevant_extra = self._has_only_irrelevant_extra()
        hasnt_markush_relevant_extra = not self._has_markush_relevant_extra()
        hasnt_markush_term = not self._has_markush_termination()
        hasnt_markush_chain = not self._has_markush_chain()
        hasnt_markush_smt_sym = not self._has_markush_smt_symbols()
        return all_sym and \
            hasnt_markush_term and \
            has_more_than_one_atom and \
            hasnt_markush_chain and \
            hasnt_markush_relevant_extra and \
            hasnt_markush_smt_sym and \
            has_irrelevant_extra 

    def is_markush(self):
        if self.rdk_mol is None:
            return False

        unmerged_fragments = Chem.GetMolFrags(self.rdk_mol)
        if len(unmerged_fragments) > 1:
            return True
        
        has_markush_sym = self._has_markush_symbols()
        has_markush_smt_sym = self._has_markush_smt_symbols()
        has_markush_term = self._has_markush_termination()
        has_markush_chain = self._has_markush_chain()
        has_markush_relevant_extra = self._has_markush_relevant_extra()
        has_brackets = self._has_brackets()
        #print([has_markush_sym, has_markush_term, has_markush_smt_sym, has_markush_chain, has_markush_relevant_extra])
        return has_markush_sym or has_markush_term or has_markush_smt_sym or has_markush_chain  or has_markush_relevant_extra or has_brackets

    def is_abbreviated(self):
        """
        from file US20210017156A1-20210121-C00032.MOL
        [...]
        M  CHG  2  22   1  23  -1
        M  STY  1   1 SUP
        M  SLB  1   1   1
        M  SAL   1  7  15  16  17  18  19  20  21
        M  SBL   1  1  22
        M  SMT   1 Boc
        M  SBV   1  22   -0.7145   -0.4125
        M  STY  1   2 SUP
        M  SLB  1   2   2
        M  SAL   2  2  22  23
        M  SBL   2  1  23
        M  SMT   2 N^C
        M  SBV   2  23    0.7145   -0.4125
        M  END
        """
        if self.rdk_mol is None:
            return False
    
        #hasnt_markush_term = not self._has_markush_termination()
        hasnt_markush_chain = not self._has_markush_chain()
        hasnt_markush_sym = not self._has_markush_symbols()
        hasnt_markush_smt_sym = not self._has_markush_smt_symbols()
        has_smt_prop = len(self.mol["sgroups"]) > 0
        hasnt_a_prop = not(len(self.mol["a_sections"]) > 0)
        hasnt_markush_relevant_extra = not self._has_markush_relevant_extra()
        #return hasnt_markush_relevant_extra and hasnt_a_prop and has_smt_prop and hasnt_markush_term and hasnt_markush_sym and hasnt_markush_smt_sym and hasnt_markush_chain
        return hasnt_markush_relevant_extra and hasnt_a_prop and has_smt_prop and hasnt_markush_sym and hasnt_markush_smt_sym and hasnt_markush_chain

    def _has_markush_chain(self):
        """
        Check for patterns
        C--[C--]_n C
        """

        if self.rdk_mol is None:
            return False

        if self.mol is None:
            return False
        try:
            conn = self.mol['conn']

            # Check for chain of [--]
            has_mark_chain = False

            # Collect pairs that have special bonds
            pairs = []
            for co in conn:
                bs = co['bs']
                atm1 = co['atm1']
                atm2 = co['atm2']

                # Enforce: atm1 < atm2
                if atm1 > atm2:
                    atm1, atm2 = atm2, atm1
                
                if bs in self.markush_bond_types:
                    pairs.append((atm1, atm2,))

            # First sweep, mark chains
            mark_chains = []
            for pair in pairs:
                atm1, atm2 = pair
                inserted = False
                for ichain in range(len(mark_chains)):
                    if atm1 in mark_chains[ichain]['atoms'] or atm2 in mark_chains[ichain]['atoms']:
                        mark_chains[ichain]['atoms'].add(atm1)
                        mark_chains[ichain]['atoms'].add(atm2)
                        mark_chains[ichain]['pairs'].append((atm1, atm2,))
                        inserted = True
                    
                if not inserted:
                    mark_chains.append({'atoms':set([atm1, atm2]), 'pairs': [(atm1, atm2,)] })

            # Second sweep, collapse mark chains
            for i in range(100):
                merge = {}
                for ichain in range(len(mark_chains)):
                    for jchain in range(ichain+1, len(mark_chains)):
                        if any([iatm in mark_chains[jchain]['atoms'] for iatm in mark_chains[ichain]['atoms']]):
                            merge.setdefault(ichain, [])
                            merge[ichain].append(jchain)
                    
                if len(merge) == 0:
                    break

                remove_chains = []
                for ichain, jchains in merge.items():
                    for jchain in jchains:
                        remove_chains.append(jchain)
                        mark_chains[ichain]['atoms'] = mark_chains[ichain]['atoms'].union(mark_chains[jchain]['atoms'])
                        mark_chains[ichain]['pairs'] += mark_chains[jchain]['pairs']
                        mark_chains[ichain]['pairs'] = list(set(mark_chains[ichain]['pairs']))

                # Remove
                for ichain in reversed(sorted(remove_chains)):
                    mark_chains.pop(ichain)

            for ichain in range(len(mark_chains)):
                if len(mark_chains[ichain]['atoms'])+1 >= 20:
                    # There are cases with many dotted lines we tag them as Markush 
                    return True
                elif len(mark_chains[ichain]['atoms'])+1 >= self.min_mark_chain:
                    pairs = mark_chains[ichain]['pairs']
                    g = self._create_G(pairs)
                    if True:
                        for n in set(g):
                            for ps in self._DFS(g, n):
                                has_mark_chain = len(ps) >= self.min_mark_chain
                                if has_mark_chain:
                                    return has_mark_chain
                    else:
                        all_paths = [p for ps in [self._DFS(g, n) for n in set(g)] for p in ps]
                        max_len = max([len(p) for p in all_paths])
                        has_mark_chain = max_len >= self.min_mark_chain
            return has_mark_chain
            
        except:
            return False
    
    @staticmethod
    def _create_G(edges):
        G = defaultdict(list)
        for (s,t) in edges:
            G[s].append(t)
            G[t].append(s)
        return G

    def _DFS(self, G,v,seen=None,path=None):
        if seen is None: seen = []
        if path is None: path = [v]

        seen.append(v)

        paths = []
        for t in G[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(self._DFS(G, t, seen[:], t_path))
        return paths
    
    def _has_markush_termination(self):
        """
        Check for patterns
        C-[...]
        |
        C--C--C
        or
        C-[...]
        |
        C--C--C
        |
        C
        or
        C-[...]
        |
        C--C  (terminal)
        or
        C-[...]
        |
        Y--X==C
        C: can be any atoms
        """

        if self.rdk_mol is None:
            return False

        if self.mol is None:
            return False
        
        conn = self.mol['conn']
        coords = self.mol['coord']

        # check for double wiggles
        has_mark_term_1 = False
        for atom in self.rdk_mol.GetAtoms():
            ai = atom.GetIdx()
            xyz = coords[ai]['xyz']

            atom_with_features = []

            bonds = atom.GetBonds()
            for bond in bonds:
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bi = bond.GetIdx()

                    bai = bond.GetBeginAtomIdx()
                    eai = bond.GetEndAtomIdx()
                    toai = eai if bai == ai else bai

                    atom_ = self.rdk_mol.GetAtomWithIdx(toai)
                    bonds_ = atom_.GetBonds()
                    is_terminal = len(bonds_) == 1
                    bs = conn[bi]['bs']
                    if bs in self.markush_bond_types and is_terminal:
                        atom_with_features.append(toai)

            #has_mark_term_1 = n_wiggles == 2
            if len(atom_with_features) >= 2:
                # Check at least 2 are in line
                for i_toai in atom_with_features:
                    for j_toai in atom_with_features:
                        # Symmetry
                        if j_toai >= i_toai:
                            continue
                        
                        i_xyz = coords[i_toai]['xyz']
                        j_xyz = coords[j_toai]['xyz']
                        
                        v = i_xyz-xyz
                        w = j_xyz-xyz
                        has_mark_term_1 = abs(1.0-abs(dot(v, w)/(norm(v)*norm(w)))) < 1e-5
                        if has_mark_term_1:
                            break

            if has_mark_term_1:
                break

        if has_mark_term_1:
            return has_mark_term_1

        # Check for terminal single wiggle on secondary atom or sp2
        has_mark_term_2 = False
        for atom in self.rdk_mol.GetAtoms():
            ai = atom.GetIdx()
            n_wiggles = 0
            n_bonds = len(atom.GetBonds())
            if n_bonds == 1: # Terminal atom
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        bi = bond.GetIdx()

                        bai = bond.GetBeginAtomIdx()
                        eai = bond.GetEndAtomIdx()
                        toai = eai if bai == ai else bai

                        atom_ = self.rdk_mol.GetAtomWithIdx(toai)
                        bonds_ = atom_.GetBonds()
                        is_secondary = len(bonds_) == 2 and atom.GetSymbol() in ['C']
                        is_sp2 = Chem.rdchem.BondType.DOUBLE in [b_.GetBondType() for b_ in bonds_] and atom.GetSymbol() in ['C']
                        is_tertiary_and_hetero = len(bonds_) == 3 and atom_.GetSymbol() in ['N', 'P', 'B']
                        bs = conn[bi]['bs']
                        if bs in self.markush_bond_types and (is_secondary or is_sp2 or is_tertiary_and_hetero):
                            n_wiggles += 1

            has_mark_term_2 = n_wiggles == 1 and n_bonds == 1
            if has_mark_term_2:
                break

        return has_mark_term_2
