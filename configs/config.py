import os

# Data Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 0 # Set to 0 for Windows compatibility with HF streaming, adjust for Linux

# Paths
DEFAULT_TAR_PATH = "dummy_dataset.tar"
MOVIENET_PATH = "dataset/MovieNet_valid/valid_mv_raw/"

# Hugging Face
HF_IMAGENET_ID = "imagenet-1k"
HF_PLACES365_ID = "ljnlonoljpiljm/places365-256px"
