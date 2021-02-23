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
import matplotlib.pyplot as plt
import keras
import tensorflow
import tensorflow as tf


import time
import numpy as np


batch_size_phase_two = 10
img_width = 299
img_height = 299
nb_epochs = 30
nb_val_samples = 5000

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




model = tf.keras.models.load_model("/home/kuja/Desktop/MajorProject/Temp/my_model")
print("DKYL3")
model.load_weights('/home/kuja/Desktop/MajorProject/Temp/finetuned_cnn_rnn_weights_2.hdf5')
print("DKYL4")
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

scores = model.evaluate_generator(val_generator, steps=nb_val_samples)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


############### PLOTS ###########################


'''
loss_train = model.history['train_loss']
loss_val = model.history['val_loss']
epochs = range(1,35)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''


