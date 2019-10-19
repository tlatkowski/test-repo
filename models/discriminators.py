from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class ConditionalDiscriminator:
    
    def __init__(self, vocab_size, embedding_size, max_sequence_length):
        self.img_height = 64
        self.img_width = 64
        self.num_channels = 3
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        # self.num_classes = input_params.num_classes
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
        input_text = Input(shape=self.max_sequence_length)
        input_image = Input(shape=(self.img_height, self.img_width, self.num_channels))
        
        embedded_id = layers.Embedding(self.vocab_size, self.embedding_size)(input_text)
        embedded_id = layers.Flatten()(embedded_id)
        embedded_id = layers.Dense(units=input_image.shape[1] * input_image.shape[2])(embedded_id)
        embedded_id = layers.Reshape(target_shape=(input_image.shape[1], input_image.shape[2], 1))(
            embedded_id)
        
        x = layers.Concatenate(axis=3)([input_image, embedded_id])
        
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.3)(x)
        
        # x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dropout(rate=0.3)(x)
        #
        # x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Dropout(rate=0.3)(x)
        
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.3)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(units=1000)(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Dense(units=1)(x)

        model = Model(name='discriminator', inputs=[input_text, input_image], outputs=x)
        
        return model
