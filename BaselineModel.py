from __future__ import print_function
import numpy as np
import json
from sklearn.preprocessing import LabelBinarizer
#import tensorflow.python.keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
import os.path
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
from resnets_utils import *
from keras import applications
from keras.models import Sequential,Model,load_model
from keras import optimizers
from keras.optimizers import Adam
import cv2 as cv2
import sys

class BaselineModel:

    def __init__(self):
        self.name = "baseline"
        self.directory = "models/"
        self.training_data = './training_data_min.npz'
        self.test_data = './val_data_min.npz'
        self.batch_size = 64
        self.num_classes = 11
        self.epochs = 30
        self.img_rows = 64
        self.img_cols = 64
        self.x_train = None
        self.x_test = None
        self.x_train_t = None
        self.x_test_t = None
        self.y_test = None
        self.y_train = None
        self.model = None
        self.learning_rate_reduction = None
        self.class_mapping = None

    def setupDatasets(self,training_data,test_data):
        try:
            train=np.load(training_data, allow_pickle=True)
            test=np.load(test_data, allow_pickle=True)
            self.x_train=train['data'][...,0]
            self.y_train=train['data'][...,1]
            self.x_test=test['data'][...,0]
            self.y_test=test['data'][...,1]
            encoder = LabelBinarizer()
            self.y_train = encoder.fit_transform(self.y_train)
            self.y_test=encoder.transform(self.y_test)
            
            x_train_tr=[self.crop_image(np.array(i)) for i in self.x_train]
            #b=np.array([x.ravel() for x in self.x_train.ravel()])
            #self.x_train_t=b.reshape(len(self.x_train),self.img_rows,self.img_cols,1)
            self.x_train_t=np.array(x_train_tr).reshape(len(self.x_train),self.img_rows,self.img_cols,1)
            
            #b=np.array([x.ravel() for x in self.x_test.ravel()])
            x_test_tr=[self.crop_image(np.array(i)) for i in self.x_test]
            #self.x_test_t=b.reshape(len(self.x_test),self.img_rows,self.img_rows,1)
            self.x_test_t=np.array(x_test_tr).reshape(len(self.x_test),self.img_rows,self.img_cols,1)
            
            self.class_mapping={i:str(k) for i,k in enumerate(encoder.classes_)}
            json_dump = json.dumps(self.class_mapping)
            f = open("label_classes.json","w")
            f.write(json_dump)
            f.close()
            print("Saved Label class mapping to disk. \n")

            print("Datasets set successfully. \n")
            print ("number of training examples = " + str(self.x_train_t.shape[0]))
            print ("number of test examples = " + str(self.x_test_t.shape[0]))
            print ("X_train shape: " + str(self.x_train_t.shape))
            print ("Y_train shape: " + str(self.y_train.shape))
            print ("X_test shape: " + str(self.x_test_t.shape))
            print ("Y_test shape: " + str(self.y_test.shape))
            print("")
            
        except Exception as e:
            print("Something went wrong on dataset setup")
            print(e)
            sys.exit(0)
        
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
            #self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
            self.model.summary()
            print("Model setup successful")
        except Exception as e:
            print("Something went wrong on model setup")
            print (e)
        

    def setupModel1(self):
        try:
            base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (self.img_rows,self.img_cols,1))
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.num_classes, activation= 'softmax')(x)
            self.model = Model(inputs = base_model.input, outputs = predictions)
            
            # Define the optimizer
            adam = Adam(lr=0.0001)
            
            # Compile the model
            self.model.compile(optimizer = adam , loss = "categorical_crossentropy", metrics=["accuracy"])
            
            # Set a learning rate annealer
            #self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
            self.model.summary()
            print("Model setup successful")
        except Exception as e:
            print("Something went wrong on model setup")
            print(e)


    def fitModel(self,augType=1):

# With data augmentation to prevent overfitting (accuracy 0.99286)
        if augType==1:
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
        if augType==2:
            datagen = ImageDataGenerator(
                            featurewise_center=True,  # set input mean to 0 over the dataset
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
        if augType==3:
            datagen = ImageDataGenerator(
                            featurewise_center=True,  # set input mean to 0 over the dataset
                            samplewise_center=False,  # set each sample mean to 0
                            featurewise_std_normalization=True,  # divide inputs by std of the dataset
                            samplewise_std_normalization=True,  # divide each input by its std
                            zca_whitening=False,  # apply ZCA whitening
                            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                            zoom_range = 0.1, # Randomly zoom image
                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                            horizontal_flip=False,  # randomly flip images
                            vertical_flip=False)  # randomly flip images


        print (self.x_train_t.shape)
        if augType>0: 
            datagen.fit(self.x_train_t)

        print ("batch_size :"+str(self.batch_size))
        print ("epochs :"+str(self.epochs))

        if augType==0:
            self.model.fit(self.x_train_t,self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,validation_data = (self.x_test_t,self.y_test))
        else:
            self.model.fit(datagen.flow(self.x_train_t,self.y_train, batch_size=self.batch_size),
                epochs = self.epochs,
                validation_data = (self.x_test_t,self.y_test),
                verbose = 1,
                steps_per_epoch=self.x_train_t.shape[0])
                #callbacks=[self.learning_rate_reduction])


    def saveModel(self, augType=1):
        # serialize model to JSON
        model_json_filename = self.directory + self.name + str(augType) + ".json"
        model_weights_filename = self.directory + self.name + str(augType) + ".h5"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with open(model_json_filename, "w") as json_file:
            json_file.write(self.model.to_json())
        # serialize weights to HDF5
        self.model.save_weights(model_weights_filename)
        print("Saved model to disk : %s"%(model_json_filename))
        print("Saved model weights to disk : %s"%(model_weights_filename))

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
        predicted_classes = []
        for k in np_image_array:
            imgarray=np.asarray(Image.fromarray(input_image.sub_images[9]).resize((64,64),Image.ANTIALIAS).convert('L'))
            predicted_classes.append(predict(imgarray))
        return predicted_classes
    
    def predict(imgarray):
        y_prob = model.predict(imgarray.reshape(1,64,64,1))
        y_classes = y_prob.argmax(axis=-1)
        return classes[y_classes[0]]
    
    def crop_image(self,imgarray):
        xmin = None
        for i,l in enumerate(imgarray.T):
            if sum(l) < 16320:
                xmin = max(i-1,0)
                break
        ymin = None
        for i,l in enumerate(imgarray):
            if sum(l) < 16320:
                ymin = max(i-1,0)
                break
        ymax = None
        for i,l in enumerate(imgarray[::-1].T):
            if sum(l) < 16320:
                ymax = min(64-i+1,64)
                break
        xmax = None
        for i,l in enumerate(imgarray[::-1]):
            if sum(l) < 16320:
                xmax = min(64-i+1,64)
                break
        return cv2.resize(imgarray[ymin:ymax,xmin:xmax], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
