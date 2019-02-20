import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Add, Input, Multiply, Concatenate
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import SGD
import os, glob, random
from PIL import Image
import pickle
import numpy as np

IMAGE_SIZE = 224
BATCH_SIZE = 32

def build_resnet50():
        input_tensor = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))
        ResNet50_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
        x = ResNet50_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        result = Dense(12,activation='softmax')(x)
        model = Model(inputs=ResNet50_model.input, outputs=result)
        for layer in model.layers[:139]:
                if 'BatchNormalization' in str(layer):
                        ...
                else:
                        layer.trainable = False
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        model.summary()
        return model

def build_vgg16():
        input_tensor = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))
        Vgg_model = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)
        x = Vgg_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        result = Dense(12,activation='softmax')(x)
        model = Model(inputs=Vgg_model.input, outputs=result)
        for layer in model.layers[:15]:
                layer.trainable = False
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        model.summary()
        return model

model = build_resnet50()
#model = build_vgg16()
model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

NUM_TRAINING = 0
NUM_TEST = 0

classes = ['Buffon','Meunier','Marquinhos','Silva','Kurzawa','Draxler',
'Rabiot','Di_maria','Verratti','Mbappe','Neymar','Cavani']
nb_classes = len(classes)

for i in classes:
        in_dir = "../Data/AllData3/train/"+i+"/*"
        in_jpg = glob.glob(in_dir)
        NUM_TRAINING += len(in_jpg)

for i in classes:
        in_dir = "../Data/AllData3/test/"+i+"/*"
        in_jpg = glob.glob(in_dir)
        NUM_TEST += len(in_jpg)

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    '../Data/AllData3/train/',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes = classes,
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   '../Data/AllData3/test/',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   classes = classes,
   shuffle=True
)

hist = model.fit_generator(train_generator,
    steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
    epochs=300,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=NUM_TEST//BATCH_SIZE
    )

model.save('./model/PSG_resnet50.h5')