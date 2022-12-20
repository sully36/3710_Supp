"""
Importing the dataset and doing any preprocessing that is required for sending it off.

/author Jessica Sullivan
"""
import glob
import tensorflow as tf
from sklearn.utils import shuffle


def download_dataset(batch_size):
    """
    Creates the tensor datasets containing the filepaths for the ADNI dataset. Right now we are only using the AD data
    within the training data. For the final question of the assignment we will need to include the NC dataset as the
    AD data only contains the scans for the participants with Alzheimer's disease. The NC data is for the neurotypical
    data.
    :param batch_size: The size of the batches we are to separate the data into.
    :return: The dataset contain the images
    """
    # Get the AD data
    train_ds = sorted(glob.glob('./AD_NC/train/AD/*.jpeg'))
    # This next line as used for early development and training processes to make computation and time not as heavy.
    # train_ds = train_ds[0:400:1]
    train_ds = shuffle(train_ds)
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    # convert the file names into the images that we need.
    train_ds = train_ds.map(preprocess)
    train_ds = train_ds.batch(batch_size)

    # Get the NC Data
    train_ds_NC = sorted(glob.glob('./AD_NC/train/AD/*.jpeg'))
    # This next line as used for early development and training processes to make computation and time not as heavy.
    # train_ds_NC = train_ds_NC[0:400:1]
    train_ds_NC = shuffle(train_ds_NC)
    train_ds_NC = tf.data.Dataset.from_tensor_slices(train_ds_NC)
    # convert the file names into the images that we need.
    train_ds_NC = train_ds_NC.map(preprocess)
    train_ds_NC = train_ds_NC.batch(batch_size)

    return train_ds, train_ds_NC


def preprocess(dataset):
    """
    Need to preprocess the data as all we have right now is a location of the image. Do this by
    reading the file and decoding the jpeg. We check to ensure that all the images are the same
    size. Then cast them to make sure in the same form I cast it to a float.
    :param dataset: the dataset that contains all the paths to the image.
    :return: the new datasets with the path to the image.
    """
    image = tf.io.read_file(dataset)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, (240, 240))
    image = tf.cast(image, tf.float32) / 1.
    return image
