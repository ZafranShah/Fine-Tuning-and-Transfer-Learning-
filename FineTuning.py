# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:52:27 2019

@author: zhshah
"""

from keras.layers import Flatten, Dense, Dropout
import numpy as np
from keras.applications import VGG16, mobilenet
from keras.models import Model, Sequential
from DataLoaderandGenerator import LoadImages
from TransferLearning1 import NewModel

####################Load Data #################################

trainDir= 'E:/Cat&DogDataset/kagglecatsanddogs_3367a/Classes/secondTry/TrainData'
testDir = 'E:/Cat&DogDataset/kagglecatsanddogs_3367a/Classes/secondTry/TestData'


obj= LoadImages(trainDir, testDir, (224,224), 180, 20, True)

generatorTrain, generatorTest= obj.LoadGenerator()

####################################################



model = NewModel

for layer in model.layers:
    trainable = ('block5' in layer.name or 'block4' in layer.name) 
    layer.trainable= trainable
    print ("{0}:\t{1}".format(layer.trainable, layer.name)) 
    
    
#optimizerFineTuning= Adam(lr=1e-7)
        
model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])  

history= model.fit_generator(generator= generatorTrain, epochs = 5 )