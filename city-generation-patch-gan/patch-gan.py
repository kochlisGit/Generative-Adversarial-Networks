import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations
import tensorflow_addons.optimizers as optimizers
import numpy as np
import cv2


# Loads dataset.
# Targets are the segmantation images.
# Inputs are the original images.
def load_dataset(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    return data['targets'], data['inputs']


# Defining the Patch-GAN discriminator.
def patch_discriminator(input_dim, target_dim):
    inputs = layers.Input(input_dim)
    targets = layers.Input(target_dim)
    merged_inputs = layers.concatenate([inputs, targets])

    x = layers.GaussianNoise(stddev=0.2)(merged_inputs)

    x = layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.4)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.4)(x)

    x = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.4)(x)

    out = layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    model = models.Model([inputs, targets], out)
    return model


# Defining U-Net3+ encoding block.
def encoder_block(inputs, n_filters, kernel_size, strides):
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    return encoder


# Defining the upscale blocks of the decoder.
def upscale_blocks(inputs):
    n_upscales = len(inputs)
    upscale_layers = []

    for i, inp in enumerate(inputs):
        p = n_upscales - i
        u = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='same')(inp)

        for i in range(2):
            u = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(u)
            u = layers.BatchNormalization()(u)
            u = layers.Activation(activations.gelu)(u)

        u = layers.Dropout(rate=0.2)(u)
        upscale_layers.append(u)
    return upscale_layers


# Defining the decoder block with the skip layers.
def decoder_block(layers_to_upscale, inputs):
    upscaled_layers = upscale_blocks(layers_to_upscale)

    decoder_blocks = []

    for i, inp in enumerate(inputs):
        d = layers.Conv2D(filters=64, kernel_size=3, strides=2**i, padding='same', use_bias=False)(inp)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)
        d = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)

        decoder_blocks.append(d)

    decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    decoder = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activations.gelu)(decoder)
    decoder = layers.Dropout(rate=0.4)(decoder)

    return decoder


# Defining the U-Net3+ generator.
def unet3_plus_generator(input_dim):
    inputs = layers.Input(input_dim)

    # Down-sampling
    e1 = encoder_block(inputs, n_filters=32, kernel_size=3, strides=2)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])

    output = layers.Conv2DTranspose(filters=3, kernel_size=1, strides=2, padding='same', activation='tanh')(d1)

    model = models.Model(inputs, output)
    return model


# Builds the patch-gan with the U-Net3+ Generator.
def build_patch_gan(generator_model, discriminator_model, image_shape):
    discriminator_model.compile(
        optimizer=optimizers.Yogi(learning_rate=0.00025, beta1=0.5),
        loss=[losses.BinaryCrossentropy(label_smoothing=0.25)],
        loss_weights=[0.5]
    )
    discriminator_model.trainable = False

    gan_input = layers.Input(image_shape)

    gen_out = generator_model(gan_input)
    disc_out = discriminator_model([gan_input, gen_out])

    gan_model = models.Model(gan_input, [disc_out, gen_out])

    gan_model.compile(
        optimizer=optimizers.Yogi(learning_rate=0.00025, beta1=0.5),
        loss=[losses.BinaryCrossentropy(label_smoothing=0.25), 'huber'],
        loss_weights=[1, 100]
    )
    return gan_model


# Saving the weights.
def save_model(gan, generator, discriminator):
    discriminator.trainable = False
    models.save_model(gan, 'model/gan')
    discriminator.trainable = True
    models.save_model(generator, 'model/generator')
    models.save_model(discriminator, 'model/discriminator')


# Loading GAN's weights.
def load_model():
    discriminator = models.load_model('model/discriminator')
    generator = models.load_model('model/generator')
    gan = models.load_model('model/gan')
    gan.summary()
    discriminator.summary()
    generator.summary()
    return generator, discriminator, gan


# Loading the dataset.
# x_train: Segmentation images.
# y_train: Original images.
DATASET_PATH = '../data/dataset.npz'
x_train, y_train = load_dataset(DATASET_PATH)

# Normalizing data.
x_train = (x_train - 127.5) / 127.5
y_train = (y_train - 127.5) / 127.5

# Building the model.
image_size = x_train[0].shape

generator = unet3_plus_generator(image_size)
discriminator = patch_discriminator(image_size, image_size)
gan = build_patch_gan(generator, discriminator, image_size)

gan.summary()

# Training the GAN model.
batch_size = 4
patch_size = 16
epochs = 150
discriminator_extra_steps = 3

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0])
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batches_per_epoch = x_train.shape[0] // batch_size

fake_labels = tf.zeros(shape=(batch_size, patch_size, patch_size, 1), dtype=tf.float32)
real_labels = tf.ones(shape=(batch_size, patch_size, patch_size, 1), dtype=tf.float32)

for epoch in range(epochs):
    print('\nTraining on epoch', epoch + 1)

    for i, (x_batch, y_batch) in enumerate(inputs):
        fakes_images = generator.predict(x_batch)

        discriminator_fake_loss = 0
        discriminator_real_loss = 0
        discriminator.trainable = True

        for step in range(discriminator_extra_steps):
            discriminator_fake_loss = discriminator.train_on_batch([x_batch, fakes_images], fake_labels)
            discriminator_real_loss = discriminator.train_on_batch([x_batch, y_batch], real_labels)

        discriminator.trainable = False

        generator_loss = gan.train_on_batch(x_batch, [real_labels, y_batch])[0]

        print('\rCurrent batch: {}/{} , Discriminator real loss = {} , Discriminator fake loss = {}, Generator loss = {}'.format(
            i+1,
            batches_per_epoch,
            discriminator_real_loss,
            discriminator_fake_loss,
            generator_loss), end='')

# Saving the model.
save_model(gan, generator, discriminator)

images_to_generate = 30
x_batch = x_train[0:images_to_generate]
generated_images = generator(x_batch)

for i, generated_image in enumerate(generated_images):
    image = np.uint8(generated_image * 127.5 + 127.5)
    cv2.imshow(str(i), image)

cv2.waitKey()
cv2.destroyAllWindows()
