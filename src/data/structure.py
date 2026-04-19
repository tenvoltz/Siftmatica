import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.util.logger import get_logger
from env import *  
from PIL import Image
import glob
from litemapy import Schematic

LOGGER = get_logger(__name__)

class Dataset(object):

    def __init__(self, name, logger=LOGGER):
        self.name = name
        self.path = os.path.join(DATA_PATH, name)
        self.images = []
        self.metadata = None
        self.litematica = None
        self.logger = logger

    def load_images(self):
        self.logger.info(f"Loading images from {os.path.join(self.path, IMAGE_FOLDER)}")
        image_files = glob.glob(os.path.join(self.path, IMAGE_FOLDER, '*.png'))
        for image_file in image_files:
            image = Image.open(image_file)
            self.images.append(image)

    def load_metadata(self):
        # TODO: Implement metadata loading if needed
        pass

    def load_litematica(self):
        self.logger.info(f"Loading Litematica schematics from {os.path.join(self.path, LITEMATICA_FOLDER)}")
        litematica_files = glob.glob(os.path.join(self.path, LITEMATICA_FOLDER, '*.litematic'))
        if litematica_files:
            self.litematica = Schematic.load(litematica_files[0])

def load_dataset(name, logger=LOGGER):
    dataset = Dataset(name, logger)
    dataset.load_images()
    dataset.load_metadata()
    dataset.load_litematica()
    return dataset

if __name__ == "__main__":
    dataset = load_dataset('house1')
    dataset.logger.info(f"Loaded dataset: {dataset.name}")
    dataset.logger.info(f"Number of images: {len(dataset.images)}")
    if dataset.litematica:
        dataset.logger.info("Litematica schematic loaded successfully.")
    else:
        dataset.logger.warning("No Litematica schematic found.")
