import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer
from utils import visualization

SEED = 0


class ConditionalGANTrainer(gan_trainer.GANTrainer):
    
    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            lr_generator,
            lr_discriminator,
            continue_training,
            checkpoint_step=10,
    ):
        super(ConditionalGANTrainer, self).__init__(
            batch_size,
            generator,
            discriminator,
            lr_generator,
            lr_discriminator,
            continue_training,
            checkpoint_step,
        )
    
    def train(self, dataset, epochs):
        test_batch_size = 1
        labels = np.array([4161, 749, 784, 928, 4156, 6441, 1866, 3268, 5103, 7113, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels = np.reshape(labels, newshape=(1, 49))
        # test_seed = [tf.random.normal([test_batch_size, 100]), labels]
        # test_seed = tf.random.normal([test_batch_size, 100])
        test_seed = tf.random.normal([1, 100])

        train_step = 0
        latest_checkpoint_epoch = 0
        
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            self.checkpoint.restore(latest_checkpoint)
            latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        epochs += latest_epoch
        for epoch in range(latest_epoch, epochs):
            for image_batch in dataset:
                # plt.imshow(image_batch[1][0])
                # img_to_plot = visualization.generate_and_save_images(self.generator, epoch + 1,
                #                                                      test_seed,
                #                                                      num_examples_to_display=test_batch_size)
                if train_step % 1000 == 0:
                    img_to_plot = visualization.generate_and_save_images(
                        self.generator,
                        train_step,
                        [test_seed, labels],
                        num_examples_to_display=test_batch_size,
                    )
                # visualization.min_max(self.generator, epoch + 1,
                #                       test_seed,
                #                       num_examples_to_display=test_batch_size)
                    print(train_step)
                train_step += 1
                gen_loss, dis_loss = self.train_step(image_batch)
                # print('gen loss', gen_loss)
                # print('dis loss', dis_loss)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
            
            img_to_plot = visualization.generate_and_save_images(self.generator, epoch + 1,
                                                                 [test_seed, labels],
                                                                 num_examples_to_display=test_batch_size)
            with self.summary_writer.as_default():
                tf.summary.image('test_images', np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                                 step=epoch)
            if (epoch + 1) % self.checkpoint_step == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    @tf.function
    def train_step(self, captions_images):
        real_captions, real_images = captions_images
        # _, real_images = real_images
        
        batch_size = real_images.shape[0]
        z = tf.random.normal([batch_size, 100])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # fake_images = self.generator([z, real_captions], training=True)
            fake_images = self.generator([z, real_captions], training=True)
            # fake_images = self.generator(z, training=True)
            
            real_output = self.discriminator([real_captions, real_images], training=True)
            # real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator([real_captions, fake_images], training=True)
            # fake_output = self.discriminator(fake_images, training=True)
            
            gen_loss = losses.generator_loss(fake_output)
            disc_loss = losses.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables,
        )
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
