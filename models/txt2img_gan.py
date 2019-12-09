from models import generators
from models import patch_discriminator
from trainers.conditional_gan_trainer import ConditionalGANTrainer


class Text2ImageGAN:
    
    def __init__(
            self,
            max_sentence_length,
            vocab_size,
    ):
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        embedding_size = 64
        self.num_epochs = 20
        self.generator = generators.TextToImageGenerator(
            max_sequence_length=max_sentence_length,
            embedding_size=embedding_size,
            vocab_size=vocab_size,
        )
        
        # self.discriminator = discriminators.ConditionalDiscriminator(vocab_size, embedding_size, max_sentence_length)
        self.discriminator = patch_discriminator.PatchDiscriminator(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            max_sequence_length=max_sentence_length,
        )
        self.gan_trainer = ConditionalGANTrainer(
            batch_size=8,
            generator=self.generator,
            discriminator=self.discriminator,
            lr_generator=0.0002,
            lr_discriminator=0.0002,
            continue_training=False,
        )
    
    def fit(self, dataset):
        self.gan_trainer.train(dataset, self.num_epochs)
