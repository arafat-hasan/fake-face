
def plot_generated_images(epoch, generator, examples=16, figsize=(20, 20)):
    noise = np.random.normal(0, 1, size=[examples, LATENT_DIM])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 128, 128, 3)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(examples / 4, 4, i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%02d.png' % epoch)


def save_model_to_file(gan, generator, discriminator, epoch):
    discriminator.trainable = False
    gan.save('facegan-gannet-epoch:%02d.h5' % epoch)
    discriminator.trainable = True
    generator.save('facegan-generator-epoch:%02d.h5' % epoch)
    discriminator.save('facegan-discriminator-epoch:%02d.h5' % epoch)


def load_or_create_model(models_paths, optimizer):

    if not (os.path.exists(models_paths["generator"]) and
            os.path.exists(models_paths["discriminator"]) and
            os.path.exists(models_paths["gan"])):
        print("[INFO] Compiling model...")
        generator = get_generator()
        discriminator = get_discriminator()
        gan = get_gan_network(discriminator, LATENT_DIM, generator, optimizer)
    else:
        print("[INFO] Loading model {}...".format(models_paths["generator"]))
        print("[INFO] Loading model {}...".format(
            models_paths["discriminator"]))
        print("[INFO] Loading model {}...".format(models_paths["gan"]))
        discriminator = load_model(models_paths["discriminator"])
        discriminator.trainable = False
        generator = load_model(models_paths["generator"])
        gan = load_model(models_paths["gan"])
        gan.summary()
        discriminator.summary()
        generator.summary()

    return gan, generator, discriminator
