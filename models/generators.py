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
        return self._model(inputs)
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def model(self):
        return self._model
    
    def create_model(self):
        # inputs = Input(shape=[self.max_sequence_length, self.embedding_size])
        z = Input(shape=[self.hidden_size])
        captions = Input(shape=self.max_sequence_length)
        
        embeddings = layers.Embedding(self.vocab_size, self.embedding_size)(captions)
        
        embeddings = attention.multihead_attention_model(embeddings)
        embeddings = layers.Flatten()(embeddings)
        
        embeddings = layers.Dense(units=8 * 8 * 32, use_bias=False)(embeddings)
        embeddings = layers.BatchNormalization()(embeddings)
        embeddings = layers.LeakyReLU()(embeddings)
        embeddings = layers.Reshape((8, 8, 32))(embeddings)
        
        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Reshape((8, 8, 256))(x)
        
        x = layers.Concatenate(axis=3)([x, embeddings])
        
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        # x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        #
        # x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                   activation='tanh')(x)
        model = Model(name='Generator', inputs=[z, captions], outputs=x)
        return model
