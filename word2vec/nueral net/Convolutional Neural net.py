
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Convolution2D

## initialising the Sequential class

convolute = Sequential()
convolute.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu'))
convolute.add(MaxPooling2D(pool_size= (2,2)))
convolute.add(Flatten())
convolute.add(Dense(units= 128, activation='relu'))
convolute.add(Dense(units=1, activation = 'sigmoid'))
convolute.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\data_cnn\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\data_cnn\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

convolute.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=2000)

from keras.preprocessing import image
import numpy as np

image_load = image.load_img(r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\data_cnn\single_prediction\cat_or_dog_1.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(image_load)
test_image = np.expand_dims(test_image, axis = 0)
result = convolute.predict(test_image)

if result[0][0] ==1:
        prediction = 'dog'
else:
        prediction = 'cat'


