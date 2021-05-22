import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


# Builds the generator.
def build_generator(random_normal_noise_size):
    generator = keras.models.Sequential()
    generator.add(keras.layers.Dense(units=7 * 7 * 128, activation='gelu', input_shape=[random_normal_noise_size]))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Reshape(target_shape=[7, 7, 128]))
    generator.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='gelu'))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', activation='sigmoid'))
    return generator


# Builds the discriminator.
def build_discriminator():
    discriminator = keras.models.Sequential()
    discriminator.add(keras.layers.Conv2D(
        filters=64, kernel_size=5, strides=2, padding='same', activation='gelu', input_shape=(28, 28, 1))
    )
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.4))
    discriminator.add(keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', activation='gelu'))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.Dropout(rate=0.4))
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return discriminator


# Builds the GAN model
def build_gan_model(generator, discriminator):
    discriminator.compile(optimizer=tfa.optimizers.Yogi(learning_rate=0.001), loss='binary_crossentropy')
    discriminator.trainable = False

    gan = keras.models.Sequential([generator, discriminator])
    gan.compile(optimizer=tfa.optimizers.Yogi(learning_rate=0.001), loss='binary_crossentropy')
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

# Normalizing the data.
x_train = x_train / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


random_normal_noise_size = 128
generator = build_generator(random_normal_noise_size)
discriminator = build_discriminator()
GAN = build_gan_model(generator, discriminator)


# Training: 1. DISCRIMINATOR - 2. GENERATOR
batch_size = 32
epochs = 50

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=8192)
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
num_of_batches = x_train.shape[0] // batch_size

for epoch in range(epochs):
    print('\nTraining on epoch', epoch+1)

    for i, x_batch in enumerate(inputs):
        # Training the discriminator first.
        random_noise = tf.random.normal(shape=[batch_size, random_normal_noise_size])
        fake_generated_images = generator(random_noise)
        mixed_fake_real_images = tf.concat([fake_generated_images, tf.dtypes.cast(x_batch, tf.float32)], axis=0)
        discriminator_targets = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)

        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(mixed_fake_real_images, discriminator_targets)

        # Training the generator after.
        random_noise = tf.random.normal(shape=[batch_size, random_normal_noise_size])
        generator_targets = tf.constant([[1.0]]*batch_size)

        discriminator.trainable = False
        generator_loss = GAN.train_on_batch(random_noise, generator_targets)

        print('\rCurrent batch: {}/{} , Discriminator loss = {} , Generator loss = {}'.format(
            i+1,
            num_of_batches,
            discriminator_loss,
            generator_loss
            ), end=''
        )


save_model(GAN, generator, discriminator)


# Generating digits.
digits_to_generate = 10
random_noise = tf.random.normal(shape=[digits_to_generate, random_normal_noise_size])
generated_digits = generator(random_noise)

for digit in generated_digits:
    plt.imshow(digit)
    plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 10))
for digit, ax in zip(generated_digits, axes.flatten()):
    ax.imshow(digit)
plt.tight_layout()
plt.title('Generated Digits')
plt.show()
