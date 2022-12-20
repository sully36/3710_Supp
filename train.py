"""
Training and running file for project.
/author Jessica Sullivan
"""
import tensorflow as tf
from module import train_step_vae, train_step_z, train_step_recon, build_vae
import time
from IPython import display
from dataset import download_dataset
import matplotlib.pyplot as plt

# parameters
epochs = 20
batch_size = 20
# batch_size = 128
kernel = 3
depth = 12

# input sizes
# todo: change the latent size? not sure if that is correct
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


def training_loop(dataset, encoder, decoder, vae, vae_opt, epoch):
    count = 0
    batch_losses = 0
    for image_batch in dataset:
        batch_losses += train_step_vae(image_batch, encoder, decoder, vae, vae_opt)
        count += 1
    # Produce images for the GIF as we go
    # display.clear_output(wait=True)
    generate_and_save_images(decoder, epoch, seed)
    return batch_losses / count


# train networks

def train(train_ds, vae, encoder, decoder):
    losses = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        # put train loop here
        loss = training_loop(train_ds, encoder, decoder, vae, vae_opt, epoch)

        print('Time for epoch {} (loss {}) is {} sec'.format(epoch, loss, time.time() - start))

        losses.append(loss)

    return losses


def generate_and_save_images(model, epoch, test_input):
    # Notice training is set to False
    # This is so all layers run in inference mode (batchnorm)
    predictions = model(test_input, training=False)
    plt.figure()
    plt.imshow(predictions[0, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Epoch {:04d}'.format(epoch))
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


train_ds, train_ds_nc = download_dataset(batch_size)
vae, encoder, decoder = build_vae(input_shape, z_size, latent_size, depth, kernel)
# convergence = train(train_ds, vae, encoder, decoder)
convergence = train(train_ds_nc, vae, encoder, decoder)
