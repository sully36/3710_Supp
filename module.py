"""
VAE Model
/author Jessica Sullivan
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, \
    Dense, Reshape
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras import backend as K


# define networks
# generator network
def decoder_network(input_shape, activation, depth, kernel, name='D'):
    """
    Decodes latent space into images
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


# discriminator network
# todo: check out what z_dim should be. Did i miss anything?
def encoder_network(input_shape, depth, kernel, latent_size, name='E'):
    """
    Encodes images into latent space
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
    Compute MMD (maximum-mean discrepancy) loss for the InfoVAE
    See https://arxiv.org/abs/1706.02262
    """

    # compute mmd
    def compute_kernel(x, y):
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        titles_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
        titles_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
        # compute the Gaussian
        return K.exp(-K.mean(K.square(titles_x - titles_y), axis=2) / K.cast(dim, 'float32'))

    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) + K.mean(xy_kernel)

    'so, we first get the mmd loss'
    'first, sample from random noise'
    batch_size = K.shape(latent)[0]
    latent_dim = K.int_shape(latent)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=1.)

    'calculate mmd loss'
    loss_mmd = compute_mmd(true_samples, latent)

    'Add them together, then you can get the final loss'
    return loss_mmd


def decoder_loss(y_true, y_pred):
    """
    Returns reconstruction loss as L2
    """
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)


# The following block was doing the vae loss in two steps rather then at the same time

@tf.function  # compiles function, much faster
def train_step_z(images, encoder, encoder_opt):
    with tf.GradientTape() as enc_tape:
        latent_codes = encoder(images, training=True)
        mmd_loss = encoder_loss(latent_codes)
    gradients_of_encoder = enc_tape.gradient(mmd_loss, encoder.trainable_variables)
    encoder_opt.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    return mmd_loss


@tf.function  # compiles function, much faster
def train_step_recon(images, encoder, decoder, decoder_opt):
    """
    The training step with the gradient tape (persistent). The switch allows for different training
    switch = 0 (compute all gradients and losses, default)
    switch = 1 (compute first gradient and loss)
    switch = 2 (compute second gradient and loss)
    :param decoder_opt:
    :param decoder:
    :param encoder:
    :param images:
    :return:
    """
    with tf.GradientTape() as dec_tape:
        latent_codes = encoder(images, training=True)
        recons = decoder(latent_codes, training=True)
        recon_loss = decoder_loss(images, recons)

    gradients_of_decoder = dec_tape.gradient(recon_loss, decoder.trainable_variables)
    decoder_opt.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))

    return recon_loss


@tf.function  # compiles function, much faster
def train_step_vae(images, encoder, decoder, vae, vae_opt):
    """
    The training step with the gradient tape (persistent). The switch allows for different training schedules.
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
