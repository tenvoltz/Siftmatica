from dotenv import load_dotenv
import os

load_dotenv()

# Get the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, os.getenv('DATA_PATH', 'data'))
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER', 'images')
LITEMATICA_FOLDER = os.getenv('LITEMATICA_FOLDER', 'litematica')
METADATA_FOLDER = os.getenv('METADATA_FOLDER', 'metadata')
STRUCTURE_FOLDER = os.getenv('STRUCTURE_FOLDER', 'structure')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'output')
