# Keras-GAN
Implementation of GAN architectures for generating digits.

# Generative Adversarial Network (GAN)
GAN is a machine learning model architecture, which is used to generate images, inspired by the Minimax theory. Two neural networks (The generator & The discriminator) contest with each other. The generator, tries to fool the discriminator, by generating random fake images, without having any knowledge of the target images. The goal of the discriminator is to classify whether an image is real or fake. An image is considered as real if it comes from the dataset. An image is considered as fake if it is generated.

# GAN Architectures in this Repository

1. Simple GAN
2. DC-GAN
3. Improved DC-GAN
4. W-GAN
5. AC-GAN
6. Pix2Pix

# Comparison of GAN models in MNIST Digit generation dataset:

# GAN

https://arxiv.org/pdf/1406.2661.pdf

Uses Denses layers both in generator & discriminator model. There are 2 major problems of GANS:
1. Mode Collapse: The generator learns simple features to fool the discriminator and exploits them.
2. Blurry Images

![GAN](https://github.com/kochlisGit/Handwritten-Digit-Generation/blob/main/mnist-digits-gan/plots/generated_digits.png)


# DC-GAN

https://arxiv.org/pdf/1511.06434v2.pdf

Uses Convolutionals (CNN) neural networks. The generator uses Conv2DTranspose layers to upscale the image, while the discriminator uses Conv2D with Strides to downscale the images.

![DC-GAN](https://github.com/kochlisGit/Handwritten-Digit-Generation/blob/main/mnist-digits-dcgan/plots/dcgan_plot.png)

# Improved DC-GAN

https://arxiv.org/pdf/1801.09195.pdf

The improved model of the DC-GAN contains methods for improving the discriminator model. This makes it harder for the generator to fool the discriminator, which forces it to learn more features and improve image quality. Some of the improvements are:
1. Adding Dropout Layers both in Generator & Discriminator. Dropout layers act as Noise, which aim to prevent **mode collapsing**.
2. Adding additional Gaussian Noise to the input of discriminator.
3. Adding Batch Normalization at the output of each layer.
4. Replacing RELU with Leaky Relu.
5. Replacing Sigmoid with Tanh activation function at the output of generator. Also, the dataset if normalized to values between -1 and 1.
6. Training the discriminator some extra steps, before training the generator.
7. Adding label smoothing in the Binary Crossentropy (BCE) loss function.
8. Setting b1 parameter of ADAM to 0.5.

![Improved DC-GAN](https://github.com/kochlisGit/Handwritten-Digit-Generation/blob/main/mnist-digits-improved-dcgan/plots/gan_norm_inputs_drop_extra_lbsmooth_plot_gauss_noise.png)

# WGAN-GP (Gradient Penalty)

https://arxiv.org/pdf/1704.00028v3.pdf

Replaces BCE loss of Discriminator with Wasserstein loss. Also, It uses Gradient Penalty method to penalize the weights.

![W-GAN](https://github.com/kochlisGit/Handwritten-Digit-Generation/blob/main/mnist-digits-wgan/plots/wgan_digits.png)

# AC-GAN

https://arxiv.org/pdf/1610.09585.pdf

Adds label information to the GAN intpus. The generator takes as an input the labels (targets) of the images. The discriminator has 2 output layers: One for
classifying whether images are real or fake and one for classifying the label (target) of an image.

![AC-GAN](https://github.com/kochlisGit/Handwritten-Digit-Generation/blob/main/mnist-digits-acgan/plots/mnist_acgan.png)

# Pix2Pix Patch-GAN

https://arxiv.org/pdf/1611.07004v3.pdf

I have implemented the patch gan as in the paper with the following improvements:

- The discriminator is trained extra steps.
- The discriminator contains noise in its inputs.
- Unet generator is replaced by Unet3+.
