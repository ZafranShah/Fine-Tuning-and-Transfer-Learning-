# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:32:02 2019

@author: zhshah
"""

from keras.preprocessing.image import ImageDataGenerator
import os 


class LoadImages:
        
    def __init__(self, Train_Dir,Test_Dir, Image_Shape,  Image_Rotation, Batch_Size ,Image_flip=False):
        
        self.trainData = Train_Dir
        self.testData = Test_Dir
        self.imgShape= Image_Shape
        self.imgRotation= Image_Rotation
        self.batchSize=Batch_Size
        self.imgFlip= Image_flip
        
    
    def PathJoin(self, dirname, filenames):
        return [os.path.join(dirname, filename) for filename in filenames]  


    def LoadGenerator(self):
        
        dataGeneratorTrain = ImageDataGenerator(rescale =  1./255,
                                   rotation_range = self.imgRotation,
                                   width_shift_range = 0.1,
                                   height_shift_range =0.1,
                                   shear_range=0.1,
                                   zoom_range =[0.9,1.5],
                                   horizontal_flip= self.imgFlip,
                                   vertical_flip= self.imgFlip,
                                   fill_mode='nearest')

        dataGeneratorTest = ImageDataGenerator(rescale =  1./255)
        generatorTrain = dataGeneratorTrain.flow_from_directory(directory = self.trainData, 
                                                        target_size= self.imgShape,
                                                        color_mode ='rgb',
                                                        batch_size= self.batchSize,
                                                        class_mode='categorical',
                                                        shuffle=True
                                                        )

        generatorTest = dataGeneratorTest.flow_from_directory(directory = self.testData, 
                                                        target_size= self.imgShape,
                                                        color_mode ='rgb',
                                                        batch_size= self.batchSize,
                                                        shuffle=False)
        
        return generatorTrain, generatorTest
    
    def TrueClasses(self, totalclasses):
        classNames=list(totalclasses.class_indices.keys())
        return classNames
    
   
    
