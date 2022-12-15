import glob
import tensorflow as tf
from sklearn.utils import shuffle
from math import floor

"""
Create a class to download and sort the ADNI data set. We shall download the train data from the zip file, which we will
then use to split into training, test and validation data 
"""


class DataSet:

    def __init__(self):
        self.validate = None
        self.testing = None
        self.training = None
        self.download_dataset()
        # self.image_shape = (256, 256)

    def download_dataset(self):
        return None
