import os
import sys
import cv2
import numpy as np
from tkinter.constants import PROJECTING
"""
ROOT_DIR = 'path/to/Mask_RCNN/'
MRCNN_WEIGHT = 'path/to/mrcnn/weight'
INPUT_USER_IMAGE = 'path/to/user/image'
INPUT_STYLE_IMAGE = 'path/to/style/image'
OUTPUT_DIR = 'path/to/output/image/dir/'
"""
PROJECT_DIR = "."

# route for dataset
DATASET_NAME = 'white2stripe'
DATASET_DIR = os.path.join(PROJECT_DIR, DATASET_NAME)
OUTPUT_DIR = os.path.join(DATASET_DIR, 'output')

# MRCNN
# route for mrcnn
MRCNN_DIR = os.path.join(PROJECT_DIR, 'mrcnn')
MRCNN_LOG_DIR = os.path.join(MRCNN_DIR, 'logs')
MRCNN_WEIGHT = os.path.join(MRCNN_LOG_DIR, 'mask_rcnn_tshirt_0028.h5')
MRCNN_INPUT = os.path.join(DATASET_DIR, 'input', 'user.jpg')
# INPUT_STYLE_IMAGE = ROOT_DIR + 'samples/Tshirt/img/input/style.jpg'
MRCNN_OUTPUT_DIR = os.path.join(DATASET_DIR, 'segmented')
# Root directory of mrcnn
MRCNN_ROOT_DIR = os.path.abspath("./mrcnn")
# Import
sys.path.append(MRCNN_ROOT_DIR)  # To find local version of the library
from config import Config
import model as modellib, utils
from mrcnn import tshirt as ts

# Cycle GAN
# route for cyclegan
GAN_DIR = os.path.join(PROJECT_DIR, 'cyclegan')
GAN_OUTPUT_DIR = os.path.join(DATASET_DIR, 'fake_output', 'user_fake.png')
# Root directory of mrcnn
GAN_ROOT_DIR = os.path.abspath("./cyclegan")
# Import
sys.path.append(GAN_ROOT_DIR)  # To find local version of the library
from cyclegan import model as cg


def main():
    #######################################
    #      Mask R-cNN(segmentation)       #
    #######################################
    class InferenceConfig(ts.TshirtConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MRCNN_LOG_DIR)
    model.load_weights(MRCNN_WEIGHT, by_name=True)
    user_fore, user_back, user_mask, user_bbox = ts.user_style_seg(MRCNN_INPUT, model, MRCNN_WEIGHT, MRCNN_OUTPUT_DIR)

    #######################################
    #               CyclaGAN              #
    #######################################
    # temp function; cannot be used. just usage example
    # generated_tshirt = CycleGAN(user_fore, style_fore, weight, ...)

    GAN = cg.CycleGAN(project_dir=PROJECT_DIR, image_folder=DATASET_NAME, to_train=False, to_restore=False, to_one=True)
    # del GAN

    #######################################
    #           Image Rendering           #
    #######################################
    # Usage: image_rendering(generated_tshirt_path, background, user_bbox, user_mask, output_dir)
    ts.image_rendering(GAN_OUTPUT_DIR + 'user_fake.jpg', user_back, user_bbox, user_mask, OUTPUT_DIR)


if __name__ == '__main__':
    import time
    t = time.process_time()
    main()
    #do some stuff
    elapsed_time = time.process_time() - t
