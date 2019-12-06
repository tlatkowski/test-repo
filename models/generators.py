import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers

from layers import attention


class TextToImageGenerator:
    
    def __init__(self, max_sequence_length, embedding_size, vocab_size):
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = 100
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
        # inputs = Input(shape=[self.max_sequence_length, self.embedding_size])
        z = Input(shape=[self.hidden_size])
        # captions = Input(shape=self.max_sequence_length)
        
        # embeddings = layers.Embedding(self.vocab_size, self.embedding_size)(captions)
        
        # embeddings = attention.multihead_attention_model(embeddings)
        # embeddings = layers.Flatten()(embeddings)
        
        # embeddings = layers.Dense(units=8 * 8 * 32, use_bias=False)(embeddings)
        # embeddings = layers.BatchNormalization()(embeddings)
        # embeddings = layers.LeakyReLU()(embeddings)
        # embeddings = layers.Reshape((8, 8, 32))(embeddings)
        
        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((8, 8, 256))(x)
        
        # x = layers.Concatenate(axis=3)([x, embeddings])
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        n_resnet = 6
        for _ in range(n_resnet):
            x = resnet_block(256, x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            activation='tanh',
        )(x)
        model = Model(name='Generator', inputs=z, outputs=x)
        # model = Model(name='Generator', inputs=[z, captions], outputs=x)
        return model


def resnet_block(n_filters, input_layer):
    g = layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        padding='same',
    )(input_layer)
    g = layers.BatchNormalization()(g)
    g = layers.ReLU()(g)
    g = layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        padding='same',
    )(g)
    g = tfa.layers.InstanceNormalization()(g)
    g = layers.Concatenate()([g, input_layer])
    return g
