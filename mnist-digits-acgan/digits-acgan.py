import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


# Builds the generator of AC-GAN.
# Includes: Dropout, Leaky Relu, Batch Normalization.
# Includes 2 Inputs: Predicting Class & Predicting Labels.
# Improvement: Replaced One-Hot Encoding vector with an Embedding Layer.
def build_generator(latent_dim, n_classes=10):
    latent_inputs = keras.Input([latent_dim])
    x1 = keras.layers.Dense(units=7 * 7 * 256, use_bias=False)(latent_inputs)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = (keras.layers.LeakyReLU()(x1))
    x1 = keras.layers.Reshape(target_shape=[7, 7, 256])(x1)

    label_inputs = keras.Input(shape=[1])
    x2 = keras.layers.Embedding(input_dim=n_classes, output_dim=64)(label_inputs)
    x2 = keras.layers.Dense(units=7*7, use_bias=False)(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.Reshape(target_shape=(7, 7, 1))(x2)

    merged_inputs = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False)(merged_inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', activation='tanh')(x)

    model = keras.Model([latent_inputs, label_inputs], x)
    return model


# Builds the discriminator of AC-GAN.
# Includes: Dropout, Leaky Relu, Batch Normalization..
# Includes 2 Models: Predicting Class & Predicting Labels.
# Improvement: Added Gaussian Noise to Inputs.
def build_discriminator(n_classes=10):
    inputs = keras.Input(shape=(28, 28, 1))

    x = keras.layers.GaussianNoise(stddev=0.2)(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.Flatten()(x)
    output1 = keras.layers.Dense(units=1, activation='sigmoid')(x)
    output2 = keras.layers.Dense(units=n_classes, activation='softmax')(x)
    model = keras.Model(inputs, [output1, output2])
    return model


# Builds the GAN model.
# Includes 2 Losses for 2 different Inputs.
# Includes Sparse BCE for target vector & BCE for Classification (Real/Fake) output.
# Improvement: Added Label Smoothing for BCE
# Improvement: Reduced Beta1 Parameter to 0.5
def build_gan(generator_model, discriminator_model):
    discriminator_model.compile(
        optimizer=tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5),
        loss=[keras.losses.BinaryCrossentropy(label_smoothing=0.25), 'sparse_categorical_crossentropy']
    )
    discriminator_model.trainable = False

    gan_input = generator_model.input
    gan_output = discriminator_model(generator_model.output)
    gan_model = keras.Model(gan_input, gan_output)
    gan_model.compile(
        optimizer=tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5),
        loss=[keras.losses.BinaryCrossentropy(label_smoothing=0.25), 'sparse_categorical_crossentropy']
    )
    return gan_model


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
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

# Normalizing the data in range [-1, 1].
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32)
x_train = (x_train - 127.5) / 127.5

# Building the model.
noise_size = 128
n_classes = 10
generator = build_generator(noise_size)
discriminator = build_discriminator()
GAN = build_gan(generator, discriminator)

# Training the GAN model.
batch_size = 32
epochs = 100
discriminator_extra_steps = 3

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0])
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batches_per_epoch = x_train.shape[0] // batch_size

real_binary_labels = tf.constant([[1.0]] * batch_size)
mixed_binary_labels = tf.constant([[1.0]] * batch_size + [[0.0]] * batch_size)

for epoch in range(epochs):
    print('\nTraining on epoch', epoch + 1)

    for i, (x_batch, y_batch) in enumerate(inputs):
        # Training the discriminator first.
        # Improvement 4. Training the discriminator more steps.
        random_noise = tf.random.normal(shape=[batch_size, noise_size])
        random_categorical_labels = np.random.randint(0, n_classes, size=[batch_size])
        fake_images = generator.predict([random_noise, random_categorical_labels])

        mixed_images = tf.concat([x_batch, fake_images], axis=0)
        mixed_categorical_labels = tf.concat([y_batch, random_categorical_labels], axis=0)

        discriminator.trainable = True
        discriminator_loss = 0
        for step in range(discriminator_extra_steps):
            discriminator_loss = discriminator.train_on_batch(
                mixed_images,
                [mixed_binary_labels, mixed_categorical_labels]
            )

        discriminator.trainable = False

        generator_loss = tf.reduce_mean(GAN.train_on_batch(
            [random_noise, random_categorical_labels],
            [real_binary_labels, random_categorical_labels])
        )

        print('\rCurrent batch: {}/{} , Discriminator loss = {} , Generator loss = {}'.format(
            i + 1,
            batches_per_epoch,
            discriminator_loss[0]/2,
            generator_loss), end='')

# Saving the model.
save_model(GAN, generator, discriminator)

digits_per_class = 3
random_noise = tf.random.normal(shape=[digits_per_class * n_classes, noise_size])
digit_targets = np.array([target for target in range(n_classes) for _ in range(digits_per_class)])
generated_digits = generator.predict([random_noise, digit_targets])

rows = 5
cols = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
for i, digit in enumerate(generated_digits):
    digit = np.reshape(digit * 127.5 + 127.5, (28, 28))
    ax = axes[i // cols, i % cols]
    ax.imshow(digit, cmap='gray')
plt.tight_layout()
plt.show()
