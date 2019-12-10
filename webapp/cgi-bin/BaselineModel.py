from __future__ import print_function
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
import os.path
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json

class BaselineModel:

    def __init__(self):
        self.name = "baseline"
        self.directory = "models/"
        self.training_data = './training_data.npz'
        self.test_data = './val_data.npz'
        self.batch_size = 86
        self.num_classes = 56
        self.epochs = 12
        self.img_rows = 100
        self.img_cols = 100
        self.x_train = None
        self.x_test = None
        self.x_train_t = None
        self.x_test_t = None
        self.y_test = None
        self.y_train = None
        self.model = None
        self.learning_rate_reduction = None

    def setupDatasets(self):
        try:
            train=np.load(self.training_data, allow_pickle=True)
            test=np.load(self.test_data, allow_pickle=True)
            self.x_train=train['data'][...,0]
            self.y_train=train['data'][...,1]
            self.x_test=test['data'][...,0]
            self.y_test=test['data'][...,1]
            encoder = LabelBinarizer()
            self.y_train = encoder.fit_transform(self.y_train)
            self.y_test=encoder.transform(self.y_test)
            b=np.array([x.ravel() for x in self.x_train.ravel()])
            self.x_train_t=b.reshape(len(self.x_train),self.img_rows,self.img_cols,1)
            b=np.array([x.ravel() for x in self.x_test.ravel()])
            self.x_test_t=b.reshape(len(self.x_test),100,100,1)
            print("Datasets set successfully")

        except:
            print("Something went wrong on dataset setup")

    def setupModel(self):
        try:
            input_shape = (self.img_rows, self.img_cols,1)

            # Define cnn model
            self.model = Sequential()
            self.model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = input_shape))
            self.model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
            self.model.add(MaxPool2D(pool_size=(2,2)))
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
            self.model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
            self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation = "relu"))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.num_classes, activation = "softmax"))

            # Define the optimizer
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

            # Compile the model
            self.model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

            # Set a learning rate annealer
            self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

            self.model.summary()
            print("Model setup successful")
        except:
            print("Something went wrong on model setup")



    def fitModel(self):

# With data augmentation to prevent overfitting (accuracy 0.99286)

        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.1, # Randomly zoom image
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        datagen.fit(self.x_train_t)

        self.model.fit(datagen.flow(self.x_train_t,self.y_train, batch_size=self.batch_size),
            epochs = self.epochs,
            validation_data = (self.x_test_t,self.y_test),
            verbose = 1,
            steps_per_epoch=self.x_train_t.shape[0],
            callbacks=[self.learning_rate_reduction])


    def saveModel(self):
        # serialize model to JSON
        model_json_filename = self.directory + self.name + ".json"
        model_weights_filename = self.directory + self.name + ".h5"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with open(model_json_filename, "w") as json_file:
            json_file.write(self.model.to_json())
        # serialize weights to HDF5
        self.model.save_weights(model_weights_filename)
        print("Saved model to disk")

    def loadModel(self):
        model_json_filename = self.directory + self.name + ".json"
        model_weights_filename = self.directory + self.name + ".h5"
        # load json and create model
        json_file = open(model_json_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(model_weights_filename)
        print("Model loaded successfully")

    def predictClasses(self,np_image_array):
        self.predicted_classes = self.model.predict_classes(np_image_array)
        return self.predicted_classes
