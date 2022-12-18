"""
Training and running file for project.
/author Jessica Sullivan
"""
import tensorflow as tf
from module import train_step_vae, train_step_z, train_step_recon, build_vae
import time
from IPython import display
from dataset import download_dataset

# parameters
epochs = 20
batch_size = 128
kernel = 3
depth = 12

# input sizes
# todo: change the latent size? not sure if that is correct
latent_size = 2
z_size = (latent_size,)
input_shape = (240, 256, 1)


# optimizers
encoder_opt = tf.keras.optimizers.Adam(1e-4)
decoder_opt = tf.keras.optimizers.Adam(1e-4)
vae_opt = tf.keras.optimizers.Adam(1e-4)

# names
model_name = "VAE-2D"


# train networks

def train(dataset, vae, encoder, decoder):
    losses = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        # put train loop here
        loss = -1
        batch_losses = 0
        switch = 1  # eppoch%3
        msg = ""
        count = 0
        for image_batch, labels_batch in dataset:
            # loss = train_step(image_batch)
            if switch == 0:  # optimise z
                loss = train_step_z(image_batch, encoder, decoder, encoder_opt)
                msg = "optimise z"
            elif switch == 2:  # optimise recon
                loss = train_step_recon(image_batch, encoder, decoder, decoder_opt)
                msg = "optimise recon"
            else:  # optimise vae
                loss = train_step_vae(image_batch, encoder, decoder, vae, vae_opt)
                msg = "optimise vae"
            batch_losses += loss
            count += 1
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        # generate_and_save_images(decoder, epoch, seed)

        loss = batch_losses / count

        print('Time for epoch {} (loss {}, {}) is {} sec'.format(epoch, loss, msg, time.time() - start))

        losses.append(loss)

    return losses


train_ds = download_dataset(batch_size)
vae, encoder, decoder = build_vae(input_shape, z_size, latent_size, depth, kernel)
convergence = train(train_ds, vae, encoder, decoder)
