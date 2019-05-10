# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:07:27 2019

@author: zhshah
"""

from keras.layers import Flatten, Dense, Dropout
import numpy as np
from keras.applications import VGG16, mobilenet
from keras.models import Model, Sequential
from DataLoaderandGenerator import LoadImages



#####################Loading Data ################################

trainDir= 'E:/Cat&DogDataset/kagglecatsanddogs_3367a/Classes/secondTry/TrainData'
testDir = 'E:/Cat&DogDataset/kagglecatsanddogs_3367a/Classes/secondTry/TestData'


obj= LoadImages(trainDir, testDir, (224,224), 180, 20, True)

generatorTrain, generatorTest= obj.LoadGenerator()
    
imagePathTrain= obj.PathJoin(trainDir,generatorTrain.filenames)
imagePathTest = obj.PathJoin(testDir, generatorTest.filenames )

classNames=obj.TrueClasses(generatorTrain)  

print (classNames)  

num_classes=len(classNames)
################################Applying Transfer Learning on Pretrained Model#####################################


def LoadPretrainedModel(model, toplayers):
    if model == 'VGG16':
        print('pretrained VGG16 model is successfully loaded ')
        VggModel = VGG16(include_top=toplayers, weights='imagenet') 
        return VggModel
    else:
        print('pretrained mobilenet model is successfully loaded ')
        mobilemodel = mobilenet.MobileNet(include_top=toplayers, weights='imagenet')
        return mobilemodel
     
        


def NewModel(generatortrain,classes, translayer, TrainedModel, model):
    transferLayer=model.get_layer(translayer)
    trainedModel= Model(inputs= model.input, outputs=transferLayer.output)
    newModel= Sequential()
    newModel.add(trainedModel)
    newModel.add(Flatten())
    newModel.add(Dense(1024, activation = 'relu'))
    newModel.add(Dropout(0.5))
    newModel.add(Dense(classes, activation = 'softmax'))
    newModel.compile(loss='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])
    for layer in trainedModel.layers:
        layer.trainable= False
        print ("{0}:\t{1}".format(layer.trainable, layer.name)) 
    step_size_train=generatortrain.n//generatortrain.batch_size
    newModel.fit_generator(generator=generatortrain,
                   steps_per_epoch=step_size_train,
                   epochs=5)
    return newModel

    
preTrainedModel=LoadPretrainedModel('VGG16', True)


modelR= NewModel(generatorTrain,num_classes,'block5_pool', LoadPretrainedModel,preTrainedModel)

