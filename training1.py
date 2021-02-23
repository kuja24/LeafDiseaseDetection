from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from keras.engine import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, LSTM
#from keras.layers import K
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras
import tensorflow
import tensorflow as tf


import time
import numpy as np

np.random.seed(1337)


class CustomImageDataGenerator(ImageDataGenerator):
    """
    Because Xception utilizes a custom preprocessing method, the only way to utilize this
    preprocessing method using the ImageDataGenerator is to overload the standardize method.

    The standardize method gets applied to each batch before ImageDataGenerator yields that batch.
    """

    def standardize(self, x):
        """
        Taken from keras.applications.xception.preprocess_input
        """
        if self.featurewise_center:
            x /= 255.
            x -= 0.5
            x *= 2.
        return x


def get_training_generator(batch_size=128):
    train_data_dir = '/home/kuja/Desktop/MajorProject/Potato/Train'
    validation_data_dir = '/home/kuja/Desktop/MajorProject/Potato/Valid'
    image_datagen = CustomImageDataGenerator(featurewise_center=True)

    train_generator = image_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size
    )
    it = iter(train_generator)
    print(next(it))

    val_generator = image_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


batch_size_phase_one = 32
batch_size_phase_two = 16
nb_val_samples = 5000


#change 30
nb_epochs = 100

img_width = 299
img_height = 299

# Setting tensorbord callback
now = time.strftime("%c")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir='./logs/' + 'cnn_rnn ' + now, histogram_freq=0, write_graph=True,
                                   write_images=False)

# Loading dataset
print("Loading the dataset with batch size of {}...".format(batch_size_phase_one))
train_generator, val_generator = get_training_generator(batch_size_phase_one)
print("Dataset loaded")

print("Building model...")
input_tensor = tf.keras.Input(shape=(img_width, img_height,3))
print("helooooo")
print(input_tensor)



# Doing Changesssssssssssssssss


'''


def build_model():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),
                                   pooling='avg')
    image_input = base_model.input
    x = base_model.layers[-1].output
    out = Dense(embedding_size)(x)
    image_embedder = Model(image_input, out)

    input_a = Input((img_size, img_size, channel), name='anchor')
    input_p = Input((img_size, img_size, channel), name='positive')
    input_n = Input((img_size, img_size, channel), name='negative')

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(input_a)
    output_a = normalize(x)
    x = image_embedder(input_p)
    output_p = normalize(x)
    x = image_embedder(input_n)
    output_n = normalize(x)

    merged_vector = concatenate([output_a, output_p, output_n], axis=-1)

    model = Model(inputs=[input_a, input_p, input_n],
                  outputs=merged_vector)
    return model 



'''

# Creating CNN
#cnn_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
channel=3
cnn_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)



#x = cnn_model.output
x = cnn_model.layers[-1].output

cnn_bottleneck = tf.keras.layers.GlobalAveragePooling2D()(x)

# Make CNN layers not trainable
for layer in cnn_model.layers:
    layer.trainable = False

# Creating RNN
x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
x = Reshape((23, 3887))(x)  # 23 timesteps, input dim of each timestep 3887
x = LSTM(2048, return_sequences=True)(x)
rnn_output = LSTM(2048)(x)

# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = tf.keras.layers.Concatenate(axis=-1)([cnn_bottleneck, rnn_output])
predictions = Dense(3, activation='softmax')(x)  

inputs2 = tf.keras.utils.get_source_inputs(input_tensor)
model = tf.keras.Model(inputs=inputs2, outputs=predictions)

print("Model built")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', verbose=1, patience=3, min_delta=0.0001)
'''
es = EarlyStopping(monitor='val_loss', verbose=1, patience=3, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


'''

print("Starting training")
checkpointer = ModelCheckpoint(filepath="/home/kuja/Desktop/MajorProject/Temp/initial_cnn_rnn_weights_2.hdf5", verbose=1,monitor='val_acc', save_best_only=True)


#change 4480
model.fit_generator(train_generator, steps_per_epoch=4480, epochs=nb_epochs, verbose=1,
                    validation_data=val_generator,
                    validation_steps=5000,
                    callbacks=[tensorboard_callback, checkpointer,es])

model.save("/home/kuja/Desktop/MajorProject/Temp/my_model")
model.save_weights('/home/kuja/Desktop/MajorProject/Temp/model.h5')

print("Initial training phase 1 done")

