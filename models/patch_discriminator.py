import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class PatchDiscriminator:
    
    def __init__(
            self,
    ):
        self.img_height = 256
        self.img_width = 256
        self.num_channels = 3
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def model(self):
        return self._model
    
    def create_model(self):
        input_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
        
        x = layers.Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
        )(input_img)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
        )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(
            filters=256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
        )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.ZeroPadding2D()(x)
        
        x = layers.Conv2D(
            filters=512,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='valid',
        )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.ZeroPadding2D()(x)
        
        x = layers.Conv2D(
            filters=1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='valid',
        )(x)
        
        model = Model(name='discriminator', inputs=input_img, outputs=x)
        
        return model
