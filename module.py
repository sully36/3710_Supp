import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, \
    Reshape
from tensorflow.keras.initializers import GlorotNormal

# parameters
epochs = 20
batch_size = 128
kernel = 3
depth = 12
# todo: change the latent size? said most cases should not be two he just used it for ease of producing plots.
latent_size = 2
z_size = (latent_size,)


# layers
def norm_conv2d(input_layer,
                n_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=ReLU(),
                use_bias=True,
                kernel_initializer=GlorotNormal(),
                **kwargs):
    """
    Create a single convolution layer with batch norm
    :param use_bias:
        todo: fill out these here:
    :param kernel_initializer:
    :param activation:
    :param input_layer:
        The input layer
    :param n_filters:
        The number of filters
    :param kernel_size:
        The size of the kernel filter
    :param strides:
        The stride number during convolution
    """
    # Create a 2D convolution layer
    conv_layer = tf.keras.layers.Conv2D(n_filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        activation=None,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        **kwargs)(input_layer)

    # adaptive batch normalization layer
    norm_layer = tf.keras.layers.BatchNormalization()(conv_layer)

    # Activation function
    layer = activation(norm_layer)
    return layer


def norm_conv2d_transpose(input_layer,
                          n_filters,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation=ReLU(),
                          use_bias=True,
                          kernel_initializer=GlorotNormal(),
                          **kwargs):
    """
    Create a single convolution transpose layer with batch norm
    :param kernel_initializer:
    :param use_bias:
    :param activation:
    :param input_layer:
        The input layer
    :param n_filters:
        The number of filters
    :param kernel_size:
        The size of the kernel filter
    :param strides:
        The stride number during convolution
    """
    # Create a 2D convolution layer
    conv_layer = tf.keras.layers.Conv2DTranspose(n_filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 activation=None,
                                                 use_bias=use_bias,
                                                 kernel_initializer=kernel_initializer,
                                                 **kwargs)(input_layer)

    # adaptive batch normalization layer
    norm_layer = tf.keras.layers.BatchNormalization()(conv_layer)

    # Activation function
    layer = activation(norm_layer)
    return layer


# define networks
# generator network
def decoder_network(input_shape, activation, name='D'):
    '''
    Decodes latent space into images
    '''
    # put you model here
    input = Input(input_shape, name=name + 'input')
    dense = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(input)
    dense = Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)

    dense = Dense(7 * 7 * depth * 2, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    dense = Reshape((7, 7, depth * 5))(dense)
    net = Conv2DTranspose(depth * 2, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(
        dense)
    net = Conv2DTranspose(depth, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(net)
    network = Conv2D(1, kernel_size=[1, 1], strides=1, activation=activation)(net)
    return Model(inputs=input, outputs=network, name=name)


# discriminator network
def encoder_network(input_shape, z_dim, name='E'):
    '''
    Encodes images into latent space
    '''
    # put you model here
    input = Input(input_shape, name=name + 'input')
    # could use stride of 1 then use a max pooling but shakes prefers strides.
    net = Conv2D(depth, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(input)
    net = Conv2D(depth * 2, kernel_size=kernel, padding='same', strides=2, activation=LeakyReLU(alpha=0.1))(input)
    # the size of the image in those dataset is 28x28 image. Then we down sampled by a factor of 2 which makes it a
    # 14x14. that was why we could do another down sampling by 2 but we cant do another one as we are at 7x7.
    dense = Flatten()(net)
    dense = Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    dense = Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    # usually latent_size needs to be 32 or 128 or something. he just did 2 to make plots easy.
    latent = Dense(latent_size, kernal_initalizer=GlorotNormal())(dense)

    return Model(inputs=input, outputs=latent, name=name)
