import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


def build_generator(noise_vector_size):
    generator = keras.models.Sequential()
    generator.add(keras.layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=[noise_vector_size]))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    generator.add(keras.layers.Reshape(target_shape=[7, 7, 256]))
    generator.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=1, use_bias=False, padding='same'))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())
    generator.add(keras.layers.Dropout(rate=0.3))
    generator.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, use_bias=False, padding='same'))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())
    generator.add(keras.layers.Dropout(rate=0.3))
    generator.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', activation='tanh'))
    return generator


def build_discriminator():
    discriminator = keras.models.Sequential()
    discriminator.add(keras.layers.GaussianNoise(stddev=0.2))
    discriminator.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False,
                                          input_shape=(28, 28, 1)))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))
    discriminator.add(keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.3))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(units=1))
    return discriminator


class WGAN(tf.keras.Model):
    def __init__(self, latent_dim, gp_weight, discriminator_extra_steps):
        super(WGAN, self).__init__()

        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.discriminator_extra_steps = discriminator_extra_steps

        self.generator = build_generator(self.latent_dim)
        self.generator_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)

        self.discriminator = build_discriminator()
        self.discriminator_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)

    def save_model(self):
        tf.keras.models.save_model(self.generator, 'model/generator')
        tf.keras.models.save_model(self.discriminator, 'model/discriminator')

    # Loading GAN's weights.
    def load_model(self):
        self.discriminator = tf.keras.models.load_model('model/discriminator')
        self.discriminator.summary()
        self.generator = tf.keras.models.load_model('model/generator')
        self.generator.summary()

    # Generates random fake images from a noise vector.
    @tf.function
    def generate(self, random_z_noise):
        return self.generator(random_z_noise)

    # Predicts whether the given images are real or fake.
    @tf.function
    def discriminate(self, images):
        return self.discriminator(images)

    # Defines the generator loss.
    @tf.function
    def generator_loss(self, fake_targets):
        return -tf.math.reduce_mean(fake_targets)

    # Defines the discriminator loss.
    @tf.function
    def discriminator_loss(self, real_targets, fake_targets):
        real_loss = tf.math.reduce_mean(real_targets)
        fake_loss = tf.math.reduce_mean(fake_targets)
        return fake_loss - real_loss

    # Computes the gradient penalty.
    @tf.function
    def gradient_penalty(self, real_images, fake_images, batch_size):
        epsilon = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        difference = real_images - fake_images
        interpolated_images = fake_images + epsilon * difference

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            interpolated_targets = self.discriminate(interpolated_images)

        gradients = gp_tape.gradient(interpolated_targets, [interpolated_images])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # Computes the total loss with the gradient penalty.
    @tf.function
    def regularized_loss(self, loss, gp):
        return loss + gp * self.gp_weight

    # Trains the model by computing the gradients and applying them to optimizers.
    @tf.function
    def train(self, x_batch, batch_size):
        z_sample = tf.random.normal(shape=[batch_size, self.latent_dim])

        for i in range(self.discriminator_extra_steps):
            with tf.GradientTape() as tape:
                fake_images = self.generate(z_sample)
                fake_targets = self.discriminate(fake_images)

                real_targets = self.discriminate(x_batch)

                gp = self.gradient_penalty(x_batch, fake_images, batch_size)
                loss = self.discriminator_loss(real_targets, fake_targets)
                discriminator_loss = self.regularized_loss(loss, gp)

                discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(
                    zip(discriminator_gradients, self.discriminator.trainable_variables)
                )

        with tf.GradientTape() as tape:
            fake_images = self.generate(z_sample)
            fake_targets = self.discriminate(fake_images)

            generator_loss = self.generator_loss(fake_targets)

            generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(
                zip(generator_gradients, self.generator.trainable_variables)
            )

        return {'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss}


# Loading the MNIST Digits dataset.
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalizing the data in range [-1, 1].
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32)
x_train = (x_train - 127.5) / 127.5
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype(np.float32)
x_test = (x_test - 127.5) / 127.5

# Building the model.
latent_dim = 128
gp_weight = 10.0
discriminator_extra_steps = 3
wgan = WGAN(latent_dim, gp_weight, discriminator_extra_steps)

# Training the GAN model.
batch_size = 64
epochs = 100
discriminator_extra_steps = 3

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=x_train.shape[0])
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batches_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    print('\nTraining on epoch', epoch + 1)

    for c, x_batch in enumerate(inputs):
        loss = wgan.train(x_batch, batch_size)

        print('\rCurrent batch: {}/{} , Discriminator loss = {} , Generator loss = {}'.format(
            c+1,
            batches_per_epoch,
            loss['discriminator_loss'],
            loss['generator_loss']), end='')

# Saving the model.
wgan.save_model()


# Generating digits.
digits_to_generate = 25
z_sample = tf.random.normal(shape=[digits_to_generate, latent_dim])
generated_digits = wgan.generate(z_sample)

rows = 5
cols = 5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
for i, digit in enumerate(generated_digits):
    ax = axes[i // rows, i % cols]
    ax.imshow(digit * 127.5 + 127.5, cmap='gray')
plt.tight_layout()
plt.show()
