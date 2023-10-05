#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import numpy as np

# Python Standard Library
import json 
import tempfile
import subprocess
import os
import logging
logger = logging.getLogger("ocsr_logger")
import uuid

# Other
from cairosvg import svg2png
from PIL import Image

# Modules
from mol_depict.utils.utils_molecule import get_molecule_from_smiles

PROJECT_PATH = os.path.dirname(__file__) + "/../../"


class CDKGenerator:
    def __init__(self):
        return
    
    def _clean_files(self, matches_json_filename, image_filename_svg, identifier, image_submolecules_matches):
        if os.path.exists(matches_json_filename):
            os.remove(matches_json_filename)
        if os.path.exists(image_filename_svg):
            os.remove(image_filename_svg)
        for label in image_submolecules_matches.keys():
            nb_instances = len(image_submolecules_matches[label])
            for i in range(nb_instances):
                if os.path.exists(PROJECT_PATH + "/external/masks/" + str(identifier) + "_" + str(label) + "_" + str(i) + ".svg"):
                    os.remove(PROJECT_PATH + "/external/cdk-depictor/masks/" + str(identifier) + "_" + str(label) + "_" + str(i) + ".svg")
        
    def generate_markush_image_masks_labels(self, smiles, image_submolecules_matches, cid):
        CDK_DEPICTOR_PATH = "/mnt/nvme/Lucas-Morin/optical-chemical-structure-recognition/external/cdk-depictor"

        #molecule = get_molecule_from_smiles(smiles, remove_stereochemistry=False)

        identifier = uuid.uuid4().hex
        matches_json_filename = CDK_DEPICTOR_PATH + "/matches_" + str(identifier) + ".json"
        with open(matches_json_filename, "w") as outfile:
            json.dump(image_submolecules_matches, outfile)

        class_path = CDK_DEPICTOR_PATH + '/project/bin:' + CDK_DEPICTOR_PATH + '/cdk-2.5.jar:' + CDK_DEPICTOR_PATH + '/gson-2.6.2.jar:' + CDK_DEPICTOR_PATH + '/json-simple-1.1.jar:.'
        java_command = 'java -cp ' + class_path + ' project.examples.Generation "' + smiles + '" "' + matches_json_filename + '" "' + str(identifier) + '"'
        java_command = java_command.translate(str.maketrans({"$":  r"\$"}))
        process = subprocess.Popen("exec " + java_command, shell=True, cwd=CDK_DEPICTOR_PATH+'/project/bin/project/examples/', \
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        image_filename_svg = CDK_DEPICTOR_PATH + "/images/" + str(identifier) +".svg"
        
        try:
            outs, errors = process.communicate(timeout=15)
        except TimeoutExpired:
            logger.error("Error in CDK depiction (1)")
            logger.handlers[0].flush()
            process.kill()
            self._clean_files(matches_json_filename, image_filename_svg, identifier, image_submolecules_matches)
            return None

        if (errors != b"") or (outs != b""):
            print(smiles)
            print(errors, outs)
            logger.error("Error in CDK depiction (2)")
            logger.handlers[0].flush()
            self._clean_files(matches_json_filename, image_filename_svg, identifier, image_submolecules_matches)
            return None

        os.remove(matches_json_filename)

        # Sometimes svg2png returns an error, to investigate // TODO 
        try:
            image_file, image_filename_png = tempfile.mkstemp()

            # Convert SVG to PNG
            svg2png(url=image_filename_svg, write_to=image_filename_png, output_width=1024, output_height=1024)
            image = Image.open(image_filename_png)

            # Clean up file
            os.close(image_file)
            os.remove(image_filename_png)
            os.remove(image_filename_svg)

        except:
            logger.error("Critic error in Markush image PNG rendering")
            # Clean up file
            os.close(image_file)
            self._clean_files(matches_json_filename, image_filename_svg, identifier, image_submolecules_matches)
            return None

        # Here, the rendering is really bad. Non perfectly black regions are hidden.
        image = np.array(image, dtype=np.float32)

        # Turn any pixel that is not perfectly white to a black pixel
        image[np.any(image <= (254., 254., 254.), axis=2)] = [0., 0., 0.] 
        image = (image != 0).astype(np.float32)

        masks = []
        labels = []

        for label in image_submolecules_matches.keys():
            nb_instances = len(image_submolecules_matches[label])
            for i in range(nb_instances):
                mask_filename_svg = PROJECT_PATH + "/external/cdk-depictor/masks/" + str(identifier) + "_" + str(label) + "_" + str(i) + ".svg"

                mask_file, mask_filename_png = tempfile.mkstemp()

                # Convert SVG to PNG
                svg2png(url=mask_filename_svg, write_to=mask_filename_png, output_width=1024, output_height=1024)
                mask = Image.open(mask_filename_png)

                # Clean up file
                os.close(mask_file)
                os.remove(mask_filename_png)
                os.remove(mask_filename_svg)

                mask = np.asarray(mask)[:, :, 0]

                # Binarize mask
                mask = (mask > 0.1).astype(np.uint8)

                masks.append(mask)
                labels.append(label)

        molecule_size = 20

        return image, masks, labels, molecule_size