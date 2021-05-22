import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

# Builds the generator.
# Improvement 1: Added more Convolutional Layers.
# Improvement 2: Replaced sigmoid with Tanh.
# Improvement 3: Added Dropout layers.
def build_generator(random_normal_noise_size):
    generator = keras.models.Sequential()
    generator.add(keras.layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=[random_normal_noise_size]))
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


# Builds the discriminator.
# Improvement 6. Added Gaussian Noise in the inputs of discriminator.
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
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return discriminator


# Builds the GAN model
# Improvement 5. Added label smoothing to loss function.
def build_gan_model(generator, discriminator):
    discriminator.compile(
        optimizer=tfa.optimizers.Yogi(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.25))
    discriminator.trainable = False

    gan = keras.models.Sequential([generator, discriminator])
    gan.compile(
        optimizer=tfa.optimizers.Yogi(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.25))
    return gan


# Saving the weights.
def save_model(gan, generator, discriminator):
    discriminator.trainable = False
    tf.keras.models.save_model(gan, 'model/gan')
    discriminator.trainable = True
    tf.keras.models.save_model(generator, 'model/generator')
    tf.keras.models.save_model(discriminator, 'model/discriminator')


# Loading GAN's weights.
def load_model():
    discriminator = tf.keras.models.load_model('model/discriminator')
    generator = tf.keras.models.load_model('model/generator')
    gan = tf.keras.models.load_model('model/gan')
    gan.summary()
    discriminator.summary()
    generator.summary()
    return generator, discriminator, gan


# Loading the MNIST Digits dataset.
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalizing the data in range [-1, 1].
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32)
x_train = (x_train - 127.5) / 127.5
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype(np.float32)
x_test = (x_test - 127.5) / 127.5

# Building the model.
random_noise_size = 128
generator = build_generator(random_noise_size)
discriminator = build_discriminator()
GAN = build_gan_model(generator, discriminator)

# Training the GAN model.
batch_size = 32
epochs = 100
discriminator_extra_steps = 3

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=x_train.shape[0])
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batches_per_epoch = x_train.shape[0] // batch_size

real_labels = tf.constant([[1.0]] * batch_size)
mixed_labels = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)

for epoch in range(epochs):
    print('\nTraining on epoch', epoch + 1)

    for i, x_batch in enumerate(inputs):
        # Training the discriminator first.
        # Improvement 4. Training the discriminator more steps.
        discriminator_loss = 0
        for step in range(discriminator_extra_steps):
            discriminator.trainable = True
            random_noise = tf.random.normal(shape=[batch_size, random_noise_size])
            fake_images = generator(random_noise)
            mixed_images = tf.concat([fake_images, tf.dtypes.cast(x_batch, tf.float32)], axis=0)
            discriminator_loss = discriminator.train_on_batch(mixed_images, mixed_labels)

        # Training the generator after.
        discriminator.trainable = False
        random_noise = tf.random.normal(shape=[batch_size, random_noise_size])
        generator_loss = GAN.train_on_batch(random_noise, real_labels)

        print('\rCurrent batch: {}/{} , Discriminator loss = {} , Generator loss = {}'.format(
            i + 1,
            batches_per_epoch,
            discriminator_loss,
            generator_loss), end='')

# Saving the model.
save_model(GAN, generator, discriminator)

# Generating digits.
digits_to_generate = 25
random_noise = tf.random.normal(shape=[digits_to_generate, random_noise_size])
generated_digits = generator(random_noise)

rows = 5
cols = 5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
for i, digit in enumerate(generated_digits):
    ax = axes[i // rows, i % cols]
    ax.imshow(digit * 127.5 + 127.5, cmap='gray')
plt.tight_layout()
plt.show()
