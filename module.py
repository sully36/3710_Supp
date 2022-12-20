"""
VAE Model. This includes all the functions required to build the VAE, indluding the encoder and decoder, as well as the
loss functions for the both.
/author Jessica Sullivan
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, \
    Dense, Reshape
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras import backend as K


def decoder_network(input_shape, activation, depth, kernel, name='D'):
    """
    Generator network. Decodes latent space into images

    :param input_shape: shape that the images are contained within
    :param activation: activation to use for the network.
    :param depth: depth of the network
    :param kernel: kernel size to use for the convolutional layers
    :param name: name of the model we are creating. Default is D for decoder.
    :return: the decoder model that can be trained.
    """
    input = Input(input_shape, name=name + 'input')
    dense = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(input)
    dense = Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)

    dense = Dense(60 * 60 * depth * 2, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    dense = Reshape((60, 60, depth * 2))(dense)
    net = Conv2DTranspose(depth * 2, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(
        dense)
    net = Conv2DTranspose(depth, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(net)
    network = Conv2D(1, kernel_size=[1, 1], strides=1, activation=activation)(net)
    return Model(inputs=input, outputs=network, name=name)


def encoder_network(input_shape, depth, kernel, latent_size, name='E'):
    """
    Discriminator network. Encodes images into latent space

    :param input_shape: shape that the images are contained within
    :param depth: depth of the network
    :param kernel: kernel size to use for the convolutional layers
    :param latent_size: the size that we want the latent space to be.
    :param name: name of the model we are creating. Default is D for decoder.
    :return: the encoder model that can be trained
    """
    input = Input(input_shape, name=name + 'input')
    # could use stride of 1 then use a max pooling but shakes prefers strides (said in lecture).
    net = Conv2D(depth, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(input)
    net = Conv2D(depth * 2, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(net)
    # the size of the image in those dataset is 28x28 image. Then we down sampled by a factor of 2 which makes it a
    # 14x14. that was why we could do another down sampling by 2 but we cant do another one as we are at 7x7.
    dense = Flatten()(net)
    dense = Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    dense = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    # usually latent_size needs to be 32 or 128 or something. he just did 2 to make plots easy.
    latent = Dense(latent_size, kernel_initializer=GlorotNormal())(dense)

    return Model(inputs=input, outputs=latent, name=name)


# loss function from InfoVAE paper
def encoder_loss(latent):
    """
    Compute MMD (maximum-mean discrepancy) loss for the VAE
    See https://arxiv.org/abs/1706.02262

    :param latent: The latent space from the encoder model.
    :return: the loss calculated through performing the encoder.
    """

    def compute_kernel(x, y):
        """
        Computes the gaussian for the kernel.
        :param x: the first data set to compare
        :param y: the second dataset to compare
        :return: the Gaussian which is the kernel for the inputs x and y
        """
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        titles_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
        titles_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
        return K.exp(-K.mean(K.square(titles_x - titles_y), axis=2) / K.cast(dim, 'float32'))

    def compute_mmd(x, y):
        """
        Finds the maximum-mean discrepancy by fining the kernel between x and y, as well as the variables with itself.
        :param x: the first dataset to compare
        :param y: the second dataset to compare
        :return: the MMD for the datasets x and y
        """
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) + K.mean(xy_kernel)

    batch_size = K.shape(latent)[0]
    latent_dim = K.int_shape(latent)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=1.)

    return compute_mmd(true_samples, latent)


def decoder_loss(y_true, y_pred):
    """
    Returns reconstruction loss as L2.
    """
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)


@tf.function  # compiles function, much faster
def train_step_vae(images, encoder, decoder, vae, vae_opt):
    """
    The training step with the gradient tape (persistent).

    :param images: images to train the vae in
    :param encoder: the model for the encoder
    :param decoder: the model for the decoder
    :param vae: the model for the VAE
    :param vae_opt: the variable to set the gradients onto
    :return: The loss calculated from training this set of images.
    """
    # scaling_step = 5
    with tf.GradientTape() as vae_tape:
        # process images and compute losses etc.
        latent_codes = encoder(images, training=True)
        recons = decoder(latent_codes, training=True)
        mmd_loss = encoder_loss(latent_codes)
        recon_loss = decoder_loss(images, recons)
        loss = mmd_loss + recon_loss

    gradients_of_vae = vae_tape.gradient(loss, vae.trainable_variables)
    vae_opt.apply_gradients(zip(gradients_of_vae, vae.trainable_variables))

    return loss


def build_vae(input_shape, z_size, latent_size, depth, kernel):
    """
    Builds the model for the VAE so that it can be trained. This will combine both the models for the encoder and the
    decoder.

    :param input_shape: the shape that the images information will be stored in
    :param z_size:
    :param latent_size: size of the latent space that we want to create from the encoder.
    :param depth: the depth of the model
    :param kernel: the size of the kernel for the convolutional layers.
    :return: the model that we created for the VAE, as well as the encoder and decoder for loss calculations later on.
    """
    # build encoder
    encoder = encoder_network(input_shape, depth, kernel, latent_size)
    encoder.summary(line_length=133)

    # build decoder
    decoder = decoder_network(z_size, ReLU(), depth, kernel)
    decoder.summary(line_length=133)

    # build VAE
    input = Input(input_shape, name='vae_input')
    z = encoder(input)
    recon = decoder(z)
    vae = Model(inputs=input, outputs=recon, name='VAE')
    vae.summary(line_length=133)
    return vae, encoder, decoder
