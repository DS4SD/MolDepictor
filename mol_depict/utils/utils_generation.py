#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import math

# Python standard library
import random
import logging
logger = logging.getLogger("ocsr_logger")

import os
import json


def coin_flip(probability):
    return random.random() < probability

def get_abbreviations(generation = True):
    abbreviations = {
        "1": [],
        "2": [],
        "3": [],
        "4": []
    }
    for abbreviation, smiles in list(get_abbreviations_smiles_mapping(generation = generation).items()):
        if len(abbreviation) == 0:
            logger.debug(f"Skipping abbreviation: {abbreviation}")
            continue 
        if abbreviation in ["O(H)"]:
            logger.debug(f"Skipping abbreviation: {abbreviation}")
            continue
        if abbreviation.isdigit():
            logger.debug(f"Skipping abbreviation: {abbreviation}")
            continue
        if abbreviation[0].isdigit():
            logger.debug(f"Skipping abbreviation: {abbreviation}")
            continue
        if ":" in abbreviation:
            logger.debug(f"Skipping abbreviation: {abbreviation}")
            continue

        replace_list = []
        for i, character in enumerate(abbreviation):
            if i == 0:
                continue
            if character.isdigit():
                replace_list.append(character)

        for character in replace_list:
            abbreviation = abbreviation.replace(character, f"<sub>{character}</sub>")
        
        if str(smiles["smiles"].count("*")) in abbreviations:
            abbreviations[str(smiles["smiles"].count("*"))].append(abbreviation)

    return abbreviations

def get_abbreviations_smiles_mapping(benchmark_dataset = None, generation = False, filtered_evaluation = False):
    abbreviations_smiles_mapping = {}
    if (benchmark_dataset == "uspto-10k-abb") or generation:
        with open(os.path.dirname(__file__) + "/../data/abbreviations/uspto-10k-abb_abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    } 
                elif abbreviation in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation]["population"] += smiles["population"]

    if (benchmark_dataset == "uspto") or (benchmark_dataset == "jpo") or generation:
        with open(os.path.dirname(__file__) + "/../data/abbreviations/uspto_abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    }
                elif abbreviation in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation]["population"] += smiles["population"]
    
    if (benchmark_dataset == "clef") or generation:
        with open(os.path.dirname(__file__) + "/../data/abbreviations/clef_abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    }
                elif abbreviation in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation]["population"] += smiles["population"]
    
    if (benchmark_dataset == "jpo") or generation:
        with open(os.path.dirname(__file__) + "/../data/abbreviations/jpo_abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    }
                elif abbreviation in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation]["population"] += smiles["population"]
    
    if ((filtered_evaluation == False) and (benchmark_dataset == "clef")) or (benchmark_dataset == None) or generation: 
        # Markush Structures Abbreviations
        with open(os.path.dirname(__file__) + "/../data/abbreviations/rgroups_abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                smiles["smiles"] = smiles["smiles"][:2] # Remove CXSMILES
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    }

    if (benchmark_dataset == "curated") or (benchmark_dataset == None) or generation:
        # Manually Curated Abbreviations
        with open(os.path.dirname(__file__) + "/../data/abbreviations/abbreviations.json", "r") as file:
            abbreviations_smiles = json.load(file)
            for abbreviation, smiles in abbreviations_smiles.items():
                if abbreviation not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[abbreviation] = {
                        "smiles": smiles["smiles"][0],
                        "population": smiles["population"]
                    }
    """
    if generation:
        # OSRA
        with open(os.path.dirname(__file__) + "/../data/abbreviations/superatom.txt", "r") as file:
            for line in file.readlines():
                line = line.strip()
                if (len(line) < 2) or (line[0] == "#"):
                    continue
                line = line.split(" ")
                if line[0] not in abbreviations_smiles_mapping.keys():
                    abbreviations_smiles_mapping[line[0]] = {
                        "smiles": line[1],
                        "population": 1
                    }
    """
    return abbreviations_smiles_mapping

def get_abbreviations_with_bond_scissors():
    abbreviations = {"O-CH3": ["O", "CH<sub>3</sub>", "H<sub>3</sub>C"],
                     "O-Me": ["O", "Me", "Me"],
                     "O-Ac": ["O", "Ac", "Ac"]
                      }
    return abbreviations

def get_abbreviations_scissors():  
    abbreviations = {
        "CF3": ["CF<sub>3</sub>", "F<sub>3</sub>C"],
        "CN": ["CN", "NC"],
        "SO3-": ["SO<sub>3</sub><sup>-</sup>", "<sup>-</sup>O<sub>3</sub>S"],
        "SO3H": ["SO<sub>3</sub>H", "HO<sub>3</sub>S"],
        "SO2Me": ["SO<sub>2</sub>Me", "MeO<sub>2</sub>S"],
        "SO2-": ["SO<sub>2</sub><sup>-</sup>", "<sup>-</sup>O<sub>2</sub>S"],
        "OMe": ["OMe", "MeO"],
        "CHO": ["CHO", "OHC"],
        "COOH": ["COOH", "HOOC"],
        "CO2H": ["CO<sub>2</sub>H", "HO<sub>2</sub>C"],
        "COO-": ["COO<sup>-</sup>", "<sup>-</sup>OOC"],
        "CO2-": ["CO<sub>2</sub><sup>-</sup>", "<sup>-</sup>O<sub>2</sub>C"],
        "OCH3": ["OCH<sub>3</sub>", "H<sub>3</sub>CO"],
        "COOMe": ["COOMe", "MeOOC"],
        "CO2Me": ["CO<sub>2</sub>Me", "MeO<sub>2</sub>C"],
        "OAc": ["OAc", "AcO"],
        "NHMe": ["NHMe", "MeNH"],
        "NMe2": ["NMe<sub>2</sub>", "Me<sub>2</sub>N"],
        "N(Me2)": ["N(Me<sub>2</sub>)", "(Me<sub>2</sub>)N"], 
        "NHAc": ["NHAc", "AcHN"],
        "N3": ["N<sub>3</sub>", "N<sub>3</sub>"],
        "NO2": ["NO<sub>2</sub>", "O<sub>2</sub>N"],
        "Me": ["Me", "Me"],
        "C2H5": ["C<sub>2</sub>H<sub>5</sub>", "H<sub>5</sub>C<sub>2</sub>"],
        "Et": ["Et", "Et"],
        "C3H7": ["C<sub>3</sub>H<sub>7</sub>", "H<sub>7</sub>C<sub>3</sub>"],
        "nPr": ["nPr", "nPr"],
        "Prn": ["Pr<sup>n</sup>", "Pr<sup>n</sup>"],
        "C4H9": ["C<sub>4</sub>H<sub>9</sub>", "H<sub>9</sub>C<sub>4</sub>"],
        "nBu": ["nBu", "nBu"],
        "Bun": ["Bu<sup>n</sup>", "Bu<sup>n</sup>"],
        "iPr": ["iPr", "iPr"],
        "Pri": ["Pr<sup>i</sup>", "Pr<sup>i</sup>"],
        "tBu": ["tBu", "tBu"],
        "But": ["Bu<sup>t</sup>", "Bu<sup>t</sup>"],
        "Ph": ["Ph", "Ph"], 
        "n-C3H7": ["n-C<sub>3</sub>H<sub>7</sub>", "n-C<sub>3</sub>H<sub>7</sub>"],
        "n-Pr": ["n-Pr", "n-Pr"],
        "n-C4H9": ["n-C<sub>4</sub>H<sub>9</sub>", "n-C<sub>4</sub>H<sub>9</sub>"],
        "n-Bu": ["n-Bu", "n-Bu"],
        "i-C3H7": ["i-C<sub>3</sub>H<sub>7</sub>", "i-C<sub>3</sub>H<sub>7</sub>"],
        "i-Pr": ["i-Pr", "i-Pr"],
        "t-C4H9": ["t-C<sub>4</sub>H<sub>9</sub>", "t-C<sub>4</sub>H<sub>9</sub>"],
        "t-Bu": ["t-Bu", "t-Bu"],
        "D": ["D", "D"],
        "*": ["*", "*"],
        "?": ["?", "?"]
    }
    return abbreviations

def get_abbreviations_large_charges(spacing):  
    """
    Groups like t-Bu with a dash are not in this list because the dash appearance is also enlarged. 
    """
    abbreviations = {
        "CF3": ["CF<sub>3</sub>", "F<sub>3</sub>C"],
        "CN": ["CN", "NC"],
        "SO3-": ["SO<sub>3</sub><sup>"+spacing+"-</sup>", "<sup>-"+spacing+"</sup>O<sub>3</sub>S"],
        "SO3H": ["SO<sub>3</sub>H", "HO<sub>3</sub>S"],
        "SO2Me": ["SO<sub>2</sub>Me", "MeO<sub>2</sub>S"],
        "SO2-": ["SO<sub>2</sub><sup>"+spacing+"-</sup>", "<sup>-"+spacing+"</sup>O<sub>2</sub>S"],
        "OMe": ["OMe", "MeO"],
        "CHO": ["CHO", "OHC"],
        "COOH": ["COOH", "HOOC"],
        "CO2H": ["CO<sub>2</sub>H", "HO<sub>2</sub>C"],
        "COO-": ["COO<sup>"+spacing+"-</sup>", "<sup>-"+spacing+"</sup>OOC"],
        "CO2-": ["CO<sub>2</sub><sup>"+spacing+"-</sup>", "<sup>-"+spacing+"</sup>O<sub>2</sub>C"],
        "OCH3": ["OCH<sub>3</sub>", "H<sub>3</sub>CO"],
        "COOMe": ["COOMe", "MeOOC"],
        "CO2Me": ["CO<sub>2</sub>Me", "MeO<sub>2</sub>C"],
        "OAc": ["OAc", "AcO"],
        "NHMe": ["NHMe", "MeNH"],
        "NMe2": ["NMe<sub>2</sub>", "Me<sub>2</sub>N"],
        "N(Me2)": ["N(Me<sub>2</sub>)", "(Me<sub>2</sub>)N"], 
        "NHAc": ["NHAc", "AcHN"],
        "N3": ["N<sub>3</sub>", "N<sub>3</sub>"],
        "NO2": ["NO<sub>2</sub>", "O<sub>2</sub>N"],
        "Me": ["Me", "Me"],
        "C2H5": ["C<sub>2</sub>H<sub>5</sub>", "H<sub>5</sub>C<sub>2</sub>"],
        "Et": ["Et", "Et"],
        "C3H7": ["C<sub>3</sub>H<sub>7</sub>", "H<sub>7</sub>C<sub>3</sub>"],
        "nPr": ["nPr", "nPr"],
        "Prn": ["Pr<sup>n</sup>", "Pr<sup>n</sup>"],
        "C4H9": ["C<sub>4</sub>H<sub>9</sub>", "H<sub>9</sub>C<sub>4</sub>"],
        "nBu": ["nBu", "nBu"],
        "Bun": ["Bu<sup>n</sup>", "Bu<sup>n</sup>"],
        "iPr": ["iPr", "iPr"],
        "Pri": ["Pr<sup>i</sup>", "Pr<sup>i</sup>"],
        "tBu": ["tBu", "tBu"],
        "But": ["Bu<sup>t</sup>", "Bu<sup>t</sup>"],
        "Ph": ["Ph", "Ph"],
        "D": ["D", "D"],
        "*": ["*", "*"],
        "?": ["?", "?"]
    }
    return abbreviations

def get_options_dict(options):
    return {
        "rotate": options.rotate,
        "fontFile": options.fontFile,
        "comicMode": options.comicMode,
        "bondLineWidth": options.bondLineWidth,
        "maxFontSize": options.maxFontSize,
        "minFontSize": options.minFontSize,
        "additionnalAtomLabelPadding": options.additionalAtomLabelPadding,
        "annotationFontScale": options.annotationFontScale,
        "addAtomIndices": options.addAtomIndices,
        "explicitMethyl": options.explicitMethyl 
    }

def get_rdkit_origin(point_1_svg, point_1_rdkit, point_2_svg, point_2_rdkit):
    """
    Compute the coordinates of the RDKit origin mapped to the svg drawing.
    """
    if abs(point_1_rdkit[0] - point_2_rdkit[0]) > 1e-2: # 1e-2 and not 0 is important!
        scale_factor = abs((point_1_svg[0] - point_2_svg[0])/(point_1_rdkit[0] - point_2_rdkit[0]))
    elif abs(point_1_rdkit[1] - point_2_rdkit[1]) > 1e-2:
        scale_factor = abs((point_1_svg[1] - point_2_svg[1])/(point_1_rdkit[1] - point_2_rdkit[1]))
    else:
        return None, None

    if scale_factor < 1e-6: # Not sure
        return None, None
        
    x_origin = point_1_svg[0] - scale_factor*(point_1_rdkit[0])
    y_origin = point_1_svg[1] + scale_factor*(point_1_rdkit[1])
    return x_origin, y_origin

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return int(qx), int(qy)
