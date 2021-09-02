from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import optimizers
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

conv_base = MobileNetV2(weights='imagenet',
                  include_top=False,
                  input_shape=(250, 250, 3))

# mobilenet architecture
conv_base.summary()


conv_base.trainable = False

mobile = models.Sequential()
mobile.add(conv_base)
mobile.add(layers.Flatten())
mobile.add(layers.Dense(256, activation='relu'))
mobile.add(layers.Dense(32, activation='relu'))
mobile.add(layers.Dense(5, activation='softmax'))

mobile.summary()

print('conv_base weight number (previous) :', len(mobile.trainable_weights))
conv_base.trainable = False
print('conv_base weights(after) :', len(mobile.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dir = 'train'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(250, 250),
    batch_size=20,
    class_mode='categorical'
    )
print(train_generator)


mobile.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['acc'])

history = mobile.fit_generator(
    train_generator,
    steps_per_epoch=15,
    epochs=50,
    validation_data=train_generator,
    validation_steps=10,
    verbose=2
)

mobile.save("food_classification_m.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()
plt.show()