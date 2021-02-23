from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint
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
batch_size_phase_two = 10
img_width = 299
img_height = 299
nb_epochs = 30
nb_val_samples = 5000

now = time.strftime("%c")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir='./logs/' + 'cnn_rnn ' + now, histogram_freq=0, write_graph=True,
                                   write_images=False)


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



def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


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

# Load two new generator with smaller batch size, needed because using the same batch size
# for the fine tuning will result in GPU running out of memory and tensorflow raising an error
print("Loading the dataset with batch size of {}...".format(batch_size_phase_two))
train_generator, val_generator = get_training_generator(batch_size_phase_two)
print("Dataset loaded")



input_tensor = tf.keras.Input(shape=(img_width, img_height,3))
print("helooooo")
print(input_tensor)


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

#model = tf.keras.models.load_model("/home/kuja/Desktop/MajorProject/Temp/my_model")
model.load_weights('/home/kuja/Desktop/MajorProject/Temp/model.h5')

# Load best weights from initial training
#model.load_weights("/home/kuja/Desktop/MajorProject/Temp/initial_cnn_rnn_weights_2.hdf5")

# Make all layers trainable for finetuning
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

print("DKYL1")
checkpointer = ModelCheckpoint(filepath="/home/kuja/Desktop/MajorProject/Temp/finetuned_cnn_rnn_weights_2.hdf5", verbose=1, save_weights_only=True)
print("DKYL2")

#change 2240
model.fit_generator(train_generator, steps_per_epoch=1500, epochs=nb_epochs, verbose=1,
                    validation_data=val_generator,
                    validation_steps=5000,
                    callbacks=[tensorboard_callback, checkpointer])

# Final evaluation of the model
print("Training done, doing final evaluation...")

print("DKYL3")
model.load_weights('/home/kuja/Desktop/MajorProject/Temp/finetuned_cnn_rnn_weights_2.hdf5')
print("DKYL4")
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

scores = model.evaluate_generator(val_generator, steps=nb_val_samples)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
