from tensorflow.keras.applications import VGG16
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import optimizers
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3))

#conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))

model.summary()

print('conv_base weight number (previous) :', len(model.trainable_weights))
conv_base.trainable = False
print('conv_base weights(after) :', len(model.trainable_weights))

train_dir = 'train'


###########################################################
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


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical'
)
print(train_generator.class_indices)
print(train_generator)


validation_dir = 'validation'

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical'
)




model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['acc'])


mc = ModelCheckpoint('display_classification_vgg1021_0001_224.h5', monitor='loss', mode='min', save_best_only=True, save_weights_only=False)     #monitor='val_loss'

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


history = model.fit(
    train_generator,
  #  steps_per_epoch=5,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[es, mc]
   # validation_steps=10,
   # verbose=2
)

#model.save('machine_classification_vgg23_0005.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training LOSS')
plt.plot(epochs, val_loss, 'b', label='Validation LOSS')
plt.legend()
plt.show()
