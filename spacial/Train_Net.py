from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import pprint

from FCN import *

# comment added

flags = tf.app.flags
flags.DEFINE_string(
    "dataset_dir", "./data/", "Dataset directory default is data/")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/",
                    "Directory name to save the checkpoints")
flags.DEFINE_string(
    "logs_path", "./logs/", "Directory name to save the log files")
flags.DEFINE_float("beta1", 0.90, "Momentum for adam")
flags.DEFINE_float("learning_rate", 0.00004, "Learning rate for adam")
flags.DEFINE_integer("batch_size", 1, "The size of the sample batch")
flags.DEFINE_integer("img_height", 300, "Image Height")
flags.DEFINE_integer("img_width", 300, "Image Width")
flags.DEFINE_float("dropout", 0.9, "Dropout")
flags.DEFINE_float("steps_per_epoch", 500, "Steps per epoch")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_boolean("continue_train", False,
                     "Continue training from previous checkpoint")
flags.DEFINE_boolean("load_Model", False, "Load Model Flag")
flags.DEFINE_string("model_path", "", "Load model from a  previous checkpoint")
flags.DEFINE_string("dataset", "CamVidV300",
                    "Choose dataset, options [Camvid, ...]")
flags.DEFINE_integer("numberClasses", 12, "Number of classes to be predicted")
flags.DEFINE_string("version_net", "FCN_Seg", "Version of the net")
flags.DEFINE_integer(
    "configuration", 4, "Set of configurations decoder [default is 4 - full decoder], other options are [1,2,3,4]")


FLAGS = flags.FLAGS

# python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS=3/
# --configuration=3


def main(_):
    
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    FCN = FCN_SS()
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_path):
        os.makedirs(FLAGS.logs_path)

    text_file = open(os.path.join(FLAGS.checkpoint_dir,'AugParam.txt'), "w")    
    
    CropAndPad_percentage = np.random.uniform(low=0.1, high=0.3)
    Flip_percentage = np.random.uniform(low=0, high=0.5)
    Scale_percentage = np.random.uniform(low=0.7, high=1)
    Rotate_degree = np.random.uniform(low=0, high=50)
    text_file.write('CropAndPad_percentage: %f \nFlip_percentage: %f \nScale_percentage: %f \nRotate_degree: %f' %(CropAndPad_percentage,Flip_percentage,Scale_percentage,Rotate_degree))
    
    text_file.close()
    # Call Train
    
    FCN.train(FLAGS, CropAndPad_percentage,Flip_percentage,Scale_percentage,Rotate_degree)


if __name__ == '__main__':
    tf.app.run()

