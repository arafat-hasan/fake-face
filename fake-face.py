#!/usr/bin/env python3
# coding: utf-8

import os
from shutil import copy, rmtree
from tqdm import tqdm
import time

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import Model
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MAIN_DATA_DIR = './celeba/celeba/'
IMAGE_DIR = './images/'

# MAIN_DATA_DIR = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'

def prepare_sample_data(nimg=1000):
    SAMPLE_DATA_DIR = './data/'
    CLASS1_DIR = './data/class-1'
    start_time = time.time()
    os.makedirs(CLASS1_DIR)
    

    for i in range(nimg):
        filename = '%05d.jpg' % (i + 1)
        src = os.path.join(MAIN_DATA_DIR, filename)
        dst = CLASS1_DIR
        copy(src, dst)

    path, dirs, files = next(os.walk(CLASS1_DIR))
    print("Total", len(files), "files copied")
    print("Data prepared in %s seconds" % (time.time() - start_time))
    return SAMPLE_DATA_DIR


def build_datagen(data_dir=None, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=5.,
        horizontal_flip=True,
    )
    
    datagen = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        color_mode='rgb',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        interpolation="bicubic",
    )
    print("[INFO] Creating data generator and loading data completed...")
    return datagen


def test_datagen(batch_size=32):
    datagen = build_datagen()
    print(len(datagen))
    for batch_count, image_batch in enumerate(datagen):
        if batch_count == 1:  # Have to use len(datagen) for all batches
            break
        if batch_count == len(datagen):
            break
        print(batch_count, image_batch.shape)
        plt.figure(figsize=(10, 10))
        for i in range(image_batch.shape[0]):
            plt.subplot(1, batch_size, i + 1)
            plt.imshow(image_batch[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()


# test_datagen(32)


class GAN():

    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        # foundation for 5x5 feature maps
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample to 8x8
        model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 16x16
        model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 64x64
        model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 128x128
        model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output layer 128x128x3
        model.add(Conv2D(3, (5, 5), activation='tanh', padding='same'))
        return model

    def build_discriminator(self):
        model = Sequential()
        # normal
        model.add(
            Conv2D(256, (5, 5), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 64x64
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 32x32
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 16x16
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 8x8
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        # opt = Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss='binary_crossentropy',
        #               optimizer=opt,
        #               metrics=['accuracy'])
        return model

    def train(self,
              epochs,
              batch_size=128,
              sample_interval=50,
              data_generator=None,
              imagepath=None):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        datagen = data_generator

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        start_time = time.time()
        for epoch in range(epochs):

            for batch_count, image_batch in tqdm(enumerate(datagen)):

                if batch_count == len(datagen):  # len(datagen)
                    break

                # ---------------------
                #  Train Discriminator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(
                    image_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples
                # as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, imagepath=imagepath)

        print("All epochs run in %s seconds" % (time.time() - start_time))

    def sample_images(self, epoch, imagepath=None, figsize=(16, 16)):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=figsize)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        plt.tight_layout()
        filename = '{}_{}.png'.format(epoch, time.strftime("%y-%m-%d_%H:%M:%S"))
        fig.savefig(os.path.join(imagepath, filename))
        plt.close()


if __name__ == '__main__':
    sample_data_dir =  prepare_sample_data(nimg=8)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    gan = GAN()
    datagen = build_datagen(sample_data_dir, batch_size=4)
    gan.train(epochs=1,
              batch_size=4,
              sample_interval=1,
              data_generator=datagen,
              imagepath=IMAGE_DIR)
    rmtree(sample_data_dir)
    print("All task done")