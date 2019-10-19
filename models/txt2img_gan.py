from models import discriminators, generators
from trainers.conditional_gan_trainer import ConditionalGANTrainer
from models import random_to_image_cifar10
from models import basic_discriminator
class Text2ImageGAN:
    
    def __init__(self, max_sentence_length, vocab_size):
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        embedding_size = 64
        self.num_epochs = 10
        # self.generator = generators.TextToImageGenerator(max_sequence_length=max_sentence_length,
        #                                                  embedding_size=embedding_size,
        #                                                  vocab_size=vocab_size)
        self.generator = random_to_image_cifar10.RandomToImageCifar10Generator()
        
        # self.discriminator = discriminators.ConditionalDiscriminator(vocab_size, embedding_size, max_sentence_length)
        self.discriminator = basic_discriminator.Discriminator()
        self.gan_trainer = ConditionalGANTrainer(64, self.generator,
                                                 self.discriminator, 0.0001, 0.0001, False)
    
    def fit(self, dataset):
        self.gan_trainer.train(dataset, self.num_epochs)
