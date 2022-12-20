"""
Training and running file for project.

/author Jessica Sullivan
"""
import tensorflow as tf
from module import train_step_vae, build_vae
import time
from dataset import download_dataset
import matplotlib.pyplot as plt

# parameters
epochs = 20
# batch_size = 20
batch_size = 400
kernel = 3
depth = 12

# input sizes
latent_size = 2
z_size = (latent_size,)
input_shape = (240, 240, 1)
seed = tf.random.normal([1, latent_size, 1])

# optimizers
encoder_opt = tf.keras.optimizers.Adam(1e-4)
decoder_opt = tf.keras.optimizers.Adam(1e-4)
vae_opt = tf.keras.optimizers.Adam(1e-4)

# names
model_name = "VAE-2D"
testing_path = "./epoch_results./"


def generate_and_save_images(model, epoch, test_input):
    """
    Generates plots of the reconstructed images after training in each epoch. This will create the image and save it but
    not open it, so that you don't have to worry about 20 images popping out at you.

    :param model: decoder that we have trained in the epoch
    :param epoch: the number of epoch we are in, so we can save it accordingly
    :param test_input: input to feed to the model
    """
    predictions = model(test_input, training=False)
    plt.figure()
    plt.imshow(predictions[0, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Epoch {:04d}'.format(epoch))
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def train(train_ds, vae, encoder, decoder):
    """
    Trains the VAE based on the batches that where created in the dataset. It will calculate the losses at every epoch
    and store those values for later use.

    :param train_ds: the dataset to train the VAE on
    :param vae: the model that was created for the VAE
    :param encoder: the model for the encoder that can be trained.
    :param decoder: the model for the decoder that can be trained.
    :return: a set recording the values of the losses over all the epochs.
    """
    losses = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        # put train loop here
        count = 0
        batch_losses = 0
        for image_batch in train_ds:
            batch_losses += train_step_vae(image_batch, encoder, decoder, vae, vae_opt)
            count += 1
        generate_and_save_images(decoder, epoch, seed)
        loss = batch_losses / count

        print('Time for epoch {} (loss {}) is {} sec'.format(epoch, loss, time.time() - start))

        losses.append(loss)

    return losses


train_ds, train_ds_nc = download_dataset(batch_size)
vae, encoder, decoder = build_vae(input_shape, z_size, latent_size, depth, kernel)
convergence = train(train_ds, vae, encoder, decoder)
# convergence = train(train_ds_nc, vae, encoder, decoder)

plt.figure()
plt.plot(range(0, epochs), convergence)
plt.grid(True)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Loss over Training')
plt.savefig('Losses.png')
