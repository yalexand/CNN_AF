# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:51:00 2022

@author: alexany
"""

import time
import sys
sys.path.insert(0,"d:\\Users\\alexany\\CNN_AF\\23_02_2022")

import numpy as np
import bioformats as bf
#import javabridge

import os
import cv2
from scipy.signal import savgol_filter

import tifffile

import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import json

#import random

import master_subFunctions as mas

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten, BatchNormalization,Activation,Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from mobilenetv3_small import MobileNetV3

#---------------------------------------------------------------------------          
class CNN_AF(object):
                
    settings_filename = 'CNN_AF_settings.json'

    image_filename = 'MMStack_Pos1.ome.tif'        
    
    stack_dir_prefix = 'stack'
    
    data_folder = None        

    zStep = 0.5  # in um
                
    n_frames = None 
                       
    image_size = 96
    
    proj_shape = None
    
    n_stacks = None
    
    inp_shape = None
               
    pList = [] # list of stack file names 
        
    #---------------------------------------------------------------------------        
    def set_data_info(self,inp_dir_list):
        self.pList = []
        for f in inp_dir_list:
            if os.path.isdir(f): 
                files = os.listdir(f)
                for i in range(len(files)):
                    if os.path.isdir(f+files[i]):
                        if self.stack_dir_prefix in files[i]:
                            fullfilename = f + files[i] + os.path.sep + self.image_filename
                            if os.path.isfile(fullfilename):
                                self.pList.append(fullfilename)
        try:            
            self.n_stacks = len(self.pList)
            imgs = tifffile.imread(self.pList[0])
            self.inp_shape = (imgs.shape[1],imgs.shape[2])                        
            self.n_frames = imgs.shape[0]
            #
            # pT, pX, pY = getImageInfo(self.pList[0])
            # self.inp_shape = (pX,pY)
            # self.n_frames = pT
            #
            self.proj_shape = (max(self.inp_shape[0],self.inp_shape[1]),2)
            #             
            # for var in vars(self):
            #     print(getattr(self,var))            
        except:
            functionNameAsString = sys._getframe().f_code.co_name
            print(functionNameAsString + ' error!')
            return False
        return True
    #---------------------------------------------------------------------------        
    def save_settings(self,where_to_save_folder):
        with open( where_to_save_folder + self.settings_filename, "w" ) as f:
            json.dump( self.__dict__, f )
    #---------------------------------------------------------------------------                    
    def load_settings(self,fullfilename):
            with open(fullfilename) as f:
                self.__dict__ = json.load(f)
    #---------------------------------------------------------------------------                   
    def get_stack_range_indices(self,k):
        beg_k = k*self.n_frames
        end_k = (k+1)*self.n_frames        

        return beg_k, end_k                
    
#---------------------------------------------------------------------------                               
class stack_data_generator(CNN_AF):    
    #---------------------------------------------------------------------------        
    def set_output_folder_in(self,dst_dir):
        self.data_folder = createFolder(dst_dir,timestamp())
        return self.data_folder
    #---------------------------------------------------------------------------        
    def gen_XY(self,index):
        X_image, X_proj, Y = 0,0,0
        #
        print(self.pList[index])         
        #
        stack     = self.load_stack(index)
        X_image   = self.gen_X_image(stack)
        X_proj    = self.gen_X_proj(stack) 
        Y         = self.gen_Y(X_image,index)                
        return X_image, X_proj, Y
    #---------------------------------------------------------------------------        
    def gen_XY_in_range(self,n1,n2):
        #
        self.save_settings(self.data_folder)
        #        
        for i in range(n1,n2):
            X_image, X_proj, Y = self.gen_XY(i)
            np.save(self.data_folder + os.path.sep + 'X_image' + '_' + str(i), X_image)
            np.save(self.data_folder + os.path.sep + 'X_proj' + '_' + str(i), X_proj)
            np.save(self.data_folder + os.path.sep + 'Y' + '_' + str(i), Y)
    #---------------------------------------------------------------------------        
    def load_stack(self,index):
        return tifffile.imread(self.pList[index])                                        
    #---------------------------------------------------------------------------        
    def gen_X_image(self,stack):
        X_image = np.zeros((int(self.n_frames),int(self.image_size), int(self.image_size))) 
        for i in range(0, self.n_frames):
            u = stack[i,:,:]
            # 
            # modified normalization - YA 25.11.2021
            # u = normalize_by_percentile(u,.1,99.9)
            # u = np.clip(u,0,1)        
            # 
            # original normalization
            u = u/np.linalg.norm(u)        
            # 
            u = cv2.resize(u, dsize=(int(self.image_size), int(self.image_size)), interpolation=cv2.INTER_CUBIC)
            X_image[i,:,:] = u
            print(i,' @ ',self.n_frames)
        return X_image
    #---------------------------------------------------------------------------        
    def gen_X_proj(self,stack):
        proj_len  = max(self.inp_shape[0],self.inp_shape[1])
        # 
        X_proj = np.zeros((int(self.n_frames),int(proj_len),int(2)))
        # 
        for i in range(0, self.n_frames):
            u = stack[i,:,:]
            # 
            # modified normalization - YA 25.11.2021
            u = normalize_by_percentile(u,.1,99.9)
            u = np.clip(u,0,1)        
            # 
            # original normalization
            # u = u/np.linalg.norm(u)        
            # 
            up1 = cv2.resize(u, dsize=(int(proj_len), int(proj_len)), interpolation=cv2.INTER_CUBIC)
            X_proj[i,:,0] = np.mean(up1,axis = 0)
            X_proj[i,:,1] = np.mean(up1,axis = 1)                        
            #
            print(i,' @ ',self.n_frames)
        return X_proj         
    #---------------------------------------------------------------------------        
    def gen_Y(self,stack,index):
    #
       xSum1 = []
       x = stack
       numSlice = self.n_frames
       #
       cgx, xSum1 = center(x, xSum1, numSlice) 
       
       yS = 0-int(cgx)
       yE = numSlice-int(cgx)
       Y = np.arange(yS, yE, 1)
       
       #print('cgx', cgx, 'ys', yS,'yE', yE,'y1',y1)
       
       #print('y0',type(y),y)
       #print('new_y', (list((np.asarray(y)-10)/25)))
       #y = ((np.asarray(y)-10.0)/25.0).tolist()
       #print('y1',type(y), y)
              
       #plt.pause(0.05)
       #plt.plot(Y,xSum1) 
       #plt.title(index)   
       
       half_range = (numSlice-1)/2.0 
       Y = ((np.asarray(Y)-0.0)/half_range).tolist()
       
       return Y
   
#---------------------------------------------------------------------------
class stack_data_trainer(CNN_AF):
    
    mode = None
    
    MobileNet = None
    #
    batch_size = 128      
    
    epos = 500     
    #
    results_folder = None
    
    train_X, valid_X, train_Y, valid_Y = None, None, None, None
    
    MobileNetV3_num_classes = int(1280)
    MobileNetV3_width_multiplier = 1.0
    MobileNetV3_l2_reg = 1e-5
    
    #---------------------------------------------------------------------------             
    def __init__(self,mode=None,MobileNet=None):
        assert(mode in ['proj','image'])
        assert(MobileNet in ['V2','V3'])
        stack_data_trainer.mode = mode
        stack_data_trainer.MobileNet = MobileNet 
    #---------------------------------------------------------------------------        
    def set_data_folder(self,data_folder):
        self.data_folder = data_folder
        try:
            self.load_settings(self.data_folder + super().settings_filename)
        except:
            print('error!')            
    #---------------------------------------------------------------------------        
    def getXY(self,validation_fraction,validation_only=False):
                    
        n_val = int(np.fix(validation_fraction*self.n_stacks))
                   
        self.valid_X, self.valid_Y = self.get_xy(0,n_val)
            
        if validation_only: return
        
        self.train_X, self.train_Y = self.get_xy(n_val,self.n_stacks)        
    #---------------------------------------------------------------------------        
    def get_xy(self,n1,n2):                    
        n = n2 - n1         
        sT = int(self.n_frames*n)
        prefix = None
        
        if self.mode=='proj':        
                sX,sY = int(self.proj_shape[0]), int(self.proj_shape[1])
                prefix = 'X_proj_'
        if self.mode=='image':                        
                sX,sY = int(self.image_size), int(self.image_size)
                prefix = 'X_image_'
            
        out_X = np.zeros((sT,sX,sY))
        out_Y = np.zeros((sT))
        for k in range(n1,n2):
            fname = self.data_folder + prefix + str(k) + '.npy'
            stack_k = np.load(fname)
            beg_k,end_k = self.get_stack_range_indices(k-n1)
            out_X[beg_k:end_k,:,:] = stack_k
            #
            fname = self.data_folder + 'Y_' + str(k) + '.npy'
            out_Y[beg_k:end_k] = np.load(fname)
            #
            print(k-n1,' @ ',n)                
            
        return out_X, out_Y  
    #---------------------------------------------------------------------------        
    def train(self,validation_fraction):
        self.results_folder = createFolder(self.data_folder,'results_' + 
                                           self.mode + '_' + 
                                           self.MobileNet + '_' + timestamp())
        self.save_settings(self.results_folder)        
        self.getXY(validation_fraction)
        #
        if self.mode=='proj':
            if self.MobileNet=='V2': 
                self.train_proj_V2()                
            if self.MobileNet=='V3':
                self.train_proj_V3()
            
        if self.mode=='image':
            if self.MobileNet=='V2': 
                self.train_image_V2()                
            if self.MobileNet=='V3':
                self.train_image_V3()                            
    #---------------------------------------------------------------------------        
    def train_proj_V2(self):            
  
        train_X = transform_stack_to_stack_of_square_images(self.train_X)
        valid_X = transform_stack_to_stack_of_square_images(self.valid_X)
        
        s = train_X.shape
        sV = valid_X.shape
        
        x_train = np.zeros([s[0], s[1], s[2], 1])
        x_val = np.zeros([sV[0], sV[1], sV[2], 1])
        
        x_train[:,:,:,0] = train_X
        x_val[:,:,:,0] = valid_X
        
        x_train_tensor = tf.convert_to_tensor(x_train)
        x_val_tensor = tf.convert_to_tensor(x_val) 
                    
        #imports the MobileNetV2 model and discards the last 1000 neuron layer.
        base_model=tf.keras.applications.MobileNetV2(weights=None,
                                                      include_top=False,
                                                      input_shape=(s[1],s[2],1)) 
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        preds = Dense(1)(x)
        model = tf.keras.Model(inputs=base_model.input,outputs=preds)  
        
        model.compile(optimizer = 'adam',
        #loss = 'sparse_categorical_crossentropy',
        loss = 'mse',
        metrics = ['mse'])
        #saveWeights(model, res_path)
        save_model_summary(self.results_folder, model)            

        #best epoch callback    
        filepath = self.results_folder+"weights_best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint] 
        
        stepsPerEpoch = int(s[0]/self.batch_size)

        history = model.fit(x_train_tensor, # Features 
                  self.train_Y, # Target vector
                  epochs=self.epos,#, # Number of epochs
                  validation_data=(x_val_tensor, self.valid_Y),
                  steps_per_epoch = stepsPerEpoch, 
                  verbose=1, # Print description after each epoch
                  #batch_size=batchSize, # Number of observations per batch
                  callbacks=callbacks_list # callbacks best model
                  #callbacks = [callback]
                  )
        #loss = history.history['loss']
        acc = history.history['mse']
        acc_val = history.history['val_mse']
        # save model
        pp = self.results_folder + "model"
        model.save(pp)
        pp = self.results_folder + "acc.npy"
        np.save(pp, acc)
        pp = self.results_folder + "acc_val.npy"
        np.save(pp, acc_val)
        #return model, history            
            
        mas.plotModelRes(self.results_folder)
        mas.save_model_summary(self.results_folder, model)
    #---------------------------------------------------------------------------    
    def train_proj_V3(self):            

        train_X = transform_stack_to_stack_of_square_images(self.train_X)
        valid_X = transform_stack_to_stack_of_square_images(self.valid_X)
        
        s = train_X.shape
        sV = valid_X.shape
        
        x_train = np.zeros([s[0], s[1], s[2], 1])
        x_val = np.zeros([sV[0], sV[1], sV[2], 1])
                    
        x_train[:,:,:,0] = train_X
        x_val[:,:,:,0] = valid_X
        
        x_train_tensor = tf.convert_to_tensor(x_train)
        x_val_tensor = tf.convert_to_tensor(x_val)  
                                        
        base_model = MobileNetV3(num_classes = self.MobileNetV3_num_classes,
                                 width_multiplier = self.MobileNetV3_width_multiplier,
                                 l2_reg = self.MobileNetV3_l2_reg)
        input_shape=(s[1],s[2],1)
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        x = base_model(input_tensor)
        x = base_model.output
        #x = GlobalAveragePooling2D()(x)
        preds = Dense(1)(x)    
        model = tf.keras.Model(inputs=[base_model.input],outputs=preds)      
                                                   
        model.compile(
            optimizer="adam",
            loss="mse",
            #loss = 'sparse_categorical_crossentropy',
            metrics=["mse"])
        
        save_model_summary(self.results_folder, model) 
        
        # best epoch callback    
        #filepath = self.results_folder+"weights_best.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        #callbacks_list = [checkpoint]              
        
        filepath = self.results_folder+"log_dir"        
        callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir=filepath)]        
                        
        stepsPerEpoch = int(s[0]/self.batch_size)
        
        history = model.fit(x_train_tensor, # Features 
                  self.train_Y, # Target vector
                  epochs=self.epos,#, # Number of epochs
                  validation_data=(x_val_tensor, self.valid_Y),
                  steps_per_epoch = stepsPerEpoch, 
                  verbose=1, # Print description after each epoch
                  #batch_size=batchSize, # Number of observations per batch
                  callbacks=callbacks_list # callbacks best model
                  )
        
        filepath = self.results_folder+"weights_best.hdf5"
        model.save_weights(filepath)
        
        #loss = history.history['loss']
        acc = history.history['mse']
        acc_val = history.history['val_mse']
        # save model
        pp = self.results_folder + "model"
        model.save(pp)
        pp = self.results_folder + "acc.npy"
        np.save(pp, acc)
        pp = self.results_folder + "acc_val.npy"
        np.save(pp, acc_val)          
        
        mas.plotModelRes(self.results_folder)
        mas.save_model_summary(self.results_folder, model) 
    #---------------------------------------------------------------------------              
    def train_image_V2(self):            
  
        s = self.train_X.shape
        sV = self.valid_X.shape
        x_train = np.zeros([s[0], s[1], s[2], 1])
        x_val = np.zeros([sV[0], sV[1], sV[2], 1])
                    
        x_train[:,:,:,0] = self.train_X
        x_val[:,:,:,0] = self.valid_X
        
        x_train_tensor = tf.convert_to_tensor(x_train)
        x_val_tensor = tf.convert_to_tensor(x_val) 
                    
        #imports the MobileNetV2 model and discards the last 1000 neuron layer.
        base_model=tf.keras.applications.MobileNetV2(weights=None,
                                                      include_top=False,
                                                      input_shape=(self.image_size,self.image_size,1)) 
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        preds = Dense(1)(x)
        model = tf.keras.Model(inputs=base_model.input,outputs=preds)  
        
        model.compile(optimizer = 'adam',
        #loss = 'sparse_categorical_crossentropy',
        loss = 'mse',
        metrics = ['mse'])
        #saveWeights(model, res_path)
        save_model_summary(self.results_folder, model)            

        #best epoch callback    
        filepath = self.results_folder+"weights_best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint] 
        
        stepsPerEpoch = int(s[0]/self.batch_size)

        history = model.fit(x_train_tensor, # Features 
                  self.train_Y, # Target vector
                  epochs=self.epos,#, # Number of epochs
                  validation_data=(x_val_tensor, self.valid_Y),
                  steps_per_epoch = stepsPerEpoch, 
                  verbose=1, # Print description after each epoch
                  #batch_size=batchSize, # Number of observations per batch
                  callbacks=callbacks_list # callbacks best model
                  #callbacks = [callback]
                  )
        #loss = history.history['loss']
        acc = history.history['mse']
        acc_val = history.history['val_mse']
        # save model
        pp = self.results_folder + "model"
        model.save(pp)
        pp = self.results_folder + "acc.npy"
        np.save(pp, acc)
        pp = self.results_folder + "acc_val.npy"
        np.save(pp, acc_val)
        #return model, history            
            
        mas.plotModelRes(self.results_folder)
        mas.save_model_summary(self.results_folder, model) 
    #---------------------------------------------------------------------------           
    def train_image_V3(self):
        
        s = self.train_X.shape
        sV = self.valid_X.shape
        x_train = np.zeros([s[0], s[1], s[2], 1])
        x_val = np.zeros([sV[0], sV[1], sV[2], 1])
                    
        x_train[:,:,:,0] = self.train_X
        x_val[:,:,:,0] = self.valid_X
        
        x_train_tensor = tf.convert_to_tensor(x_train)
        x_val_tensor = tf.convert_to_tensor(x_val)  
                                        
        base_model = MobileNetV3(num_classes = self.MobileNetV3_num_classes,
                                 width_multiplier = self.MobileNetV3_width_multiplier,
                                 l2_reg = self.MobileNetV3_l2_reg)

        input_shape=(self.image_size,self.image_size,1)
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        x = base_model(input_tensor)
        x = base_model.output
        #x = GlobalAveragePooling2D()(x)
        preds = Dense(1)(x)    
        model = tf.keras.Model(inputs=[base_model.input],outputs=preds)      
                                                   
        model.compile(
            optimizer="adam",
            loss="mse",
            #loss = 'sparse_categorical_crossentropy',
            metrics=["mse"])
        
        save_model_summary(self.results_folder, model) 
    
        # best epoch callback    
        #filepath = self.results_folder+"weights_best.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        #callbacks_list = [checkpoint]              
        
        filepath = self.results_folder+"log_dir"        
        callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir=filepath)]        
                        
        stepsPerEpoch = int(s[0]/self.batch_size)
    
        history = model.fit(x_train_tensor, # Features 
                  self.train_Y, # Target vector
                  epochs=self.epos,#, # Number of epochs
                  validation_data=(x_val_tensor, self.valid_Y),
                  steps_per_epoch = stepsPerEpoch, 
                  verbose=1, # Print description after each epoch
                  #batch_size=batchSize, # Number of observations per batch
                  callbacks=callbacks_list # callbacks best model
                  )
        
        filepath = self.results_folder+"weights_best.hdf5"
        model.save_weights(filepath)
                
        #loss = history.history['loss']
        acc = history.history['mse']
        acc_val = history.history['val_mse']
        # save model
        pp = self.results_folder + "model"
        model.save(pp)
        pp = self.results_folder + "acc.npy"
        np.save(pp, acc)
        pp = self.results_folder + "acc_val.npy"
        np.save(pp, acc_val)          
        
        mas.plotModelRes(self.results_folder)
        mas.save_model_summary(self.results_folder, model)                                                 
                          
#---------------------------------------------------------------------------
class stack_data_tester(stack_data_trainer):
    #            
    modelLoad = None
    n = None # number of actual stacks     
    #---------------------------------------------------------------------------   
    def test(self,zStep=None,validation_fraction=None,results_folder=None,token=None):
        n_val = int(np.fix(validation_fraction*self.n_stacks))
        self.test_range(zStep,0,n_val,results_folder,token)
    #---------------------------------------------------------------------------           
    def test_range(self,zStep=None,n1=0,n2=0,results_folder=None,token=None):
        #
        self.setup(n1,n2,results_folder)                               
        #
        listPred, listY, listResid = self.get_Pred_Y_Resid_for_plotting(zStep)
        #
        make_plots(listPred, listY, listResid, self.n_frames, self.results_folder, token, zStep)
        accCalc(listPred, listY, self.n_frames, self.results_folder, token, zStep)
        residMean(listResid, listY, self.n_frames, self.results_folder, token, zStep)        
    #---------------------------------------------------------------------------
    def setup(self,n1,n2,results_folder=None):
        assert None != self.data_folder
        self.results_folder = results_folder       
        self.valid_X, self.valid_Y = self.get_xy(n1,n2)  
        self.n = int(self.valid_X.shape[0]/self.n_frames)
        if 'V2'==self.MobileNet:
            self.modelLoad = tf.keras.models.load_model(self.results_folder + os.path.sep + "model")
            self.modelLoad.load_weights(self.results_folder + os.path.sep + "weights_best.hdf5")
        else:
            base_model = MobileNetV3(num_classes = self.MobileNetV3_num_classes,
                                     width_multiplier = self.MobileNetV3_width_multiplier,
                                     l2_reg = self.MobileNetV3_l2_reg)
            if 'image'==self.mode:
                input_shape=(self.image_size,self.image_size,1)
            else: 
                if 'proj'==self.mode:
                    valid_X = transform_stack_to_stack_of_square_images(self.valid_X)                    
                    s = valid_X.shape                                     
                    input_shape=(s[1],s[2],1)                                                            
            #                                     
            input_tensor = tf.keras.layers.Input(shape=input_shape)                      
            x = base_model(input_tensor)
            x = base_model.output
            #x = GlobalAveragePooling2D()(x)
            preds = Dense(1)(x)                                    
            self.modelLoad = tf.keras.Model(inputs=[base_model.input],outputs=preds)                
            self.modelLoad.compile(
                optimizer="adam",
                loss="mse",
                #loss = 'sparse_categorical_crossentropy',
                metrics=["mse"])                                                
            self.modelLoad.load_weights(self.results_folder + os.path.sep + "weights_best.hdf5")                                                
    #---------------------------------------------------------------------------               
    def get_Pred_Y_Resid_for_plotting(self,zStep):
        half_range = (self.n_frames-1)*zStep/2.0
        #                        
        listPred = []
        listY = []
        listResid = []        
        #
        for k in range(0, self.n):
            print(k)
            xVal = self.get_ground_truth(k)        
            yVal = self.get_predictions(k)
            yVal = ((np.asarray(yVal)*half_range)+0.0)
            xVal = ((np.asarray(xVal)*half_range)+0.0)                               
            resid = yVal - xVal                        
            listPred.append(yVal)
            listY.append(xVal)
            listResid.append(resid)
        #                                
        return listPred,listY,listResid
    #---------------------------------------------------------------------------           
    def get_predictions(self,k):                                
        if self.mode=='proj':
            return self.get_predictions_proj(k)                
        else:
            return self.get_predictions_image(k)
    #---------------------------------------------------------------------------                               
    def get_predictions_image(self,k):
        #
        x_val = np.zeros([self.n_frames,self.image_size,self.image_size, 1])
        beg_k, end_k = self.get_stack_range_indices(k)         
        x_val[:,:,:,0] = self.valid_X[beg_k:end_k,:,:]
        x_val_tensor = tf.convert_to_tensor(x_val)
        predictions = self.modelLoad.predict(x_val_tensor) 
        return predictions[:,0]                
    #---------------------------------------------------------------------------                               
    def get_predictions_proj(self,k):
        #
        beg_k, end_k = self.get_stack_range_indices(k) 
        x_val_ =  transform_stack_to_stack_of_square_images(self.valid_X[beg_k:end_k,:,:])        
        x_val = np.zeros([self.n_frames,x_val_.shape[1],x_val_.shape[1], 1])                
        x_val[:,:,:,0] = x_val_
        x_val_tensor = tf.convert_to_tensor(x_val)
        predictions = self.modelLoad.predict(x_val_tensor) 
        return predictions[:,0]                                
    #---------------------------------------------------------------------------                   
    def get_ground_truth(self,k):
        beg_k, end_k = self.get_stack_range_indices(k)        
        return self.valid_Y[beg_k:end_k]
    
#%%        
# Y: centering functions 
#---------------------------------------------------------------------------
def center(x, xSum1, numSlice):
    cgx, xSum1 = centerStd2(x, xSum1, numSlice)
    return cgx, xSum1
#---------------------------------------------------------------------------
def centerStd2(x, xSum1, numSlice):
    background = np.amin(x, axis=0)
    #bg_projection = np.mean(background, axis = 0)
    savgol_projections0 = []
    savgol_projections1 = []
    for ii in range(numSlice):
        
        xx = x[ii]- background
        
        mean_axis0 = np.mean(xx, axis = 0)# - bg_projection
        mean_axis1 = np.mean(xx, axis = 1)
        mean_axis0 = savgol_filter(mean_axis0, 21, 3)
        mean_axis1 = savgol_filter(mean_axis1, 21, 3)
        savgol_projections0.append(mean_axis0)
        savgol_projections1.append(mean_axis1)
    bg_projection0 = np.amin(savgol_projections0, axis=0)
    bg_projection1 = np.amin(savgol_projections1, axis=0)
    savgol_projections0 = savgol_projections0 - bg_projection0
    savgol_projections1 = savgol_projections1 - bg_projection1
    #for iii in range(len(savgol_projections1)):
    for iii in range(numSlice):
        start_time = time.time()
        xx = x[iii]- background
        max_ = np.max(xx)#/np.std(xx)
        v2 = xx >= 0.5*max_
        num_ = np.sum(v2)
        ssum1 = sum(sum(xx))
        mean_axis0 = np.mean(xx, axis = 0)# - bg_projection
        mean_axis1 = np.mean(xx, axis = 1)
        sp0 = np.fft.fft(savgol_projections0[iii])
        ps0 = np.abs(sp0)**2
        samp_length0 = np.arange(len(savgol_projections0[iii]))
        freq0 = np.fft.fftfreq(samp_length0.shape[-1])
        idx0 = np.argsort(freq0)
        ps_std0 = np.std(ps0[idx0])
        sp1 = np.fft.fft(savgol_projections1[iii])
        ps1 = np.abs(sp1)**2
        samp_length1 = np.arange(len(savgol_projections1[iii]))
        freq1 = np.fft.fftfreq(samp_length1.shape[-1])
        idx1 = np.argsort(freq1)
        ps_std1 = np.std(ps1[idx1])
        #plt.figure('ps')
        #plt.plot(freq[idx], ps[idx])
        #plt.show()
        std0 = np.std(mean_axis0)
        std1 = np.std(mean_axis1)
        std_tot = np.std(xx)
        #std_comb = std0*std1
        #std_comb = np.std(xx)
        end_time = time.time()
        #print(iii, end_time-start_time)
        #xSum1.append((ps_std0*std0)/(ps_std1*std1))
        xSum1.append(ps_std1)

    xSum1 = savgol_filter(xSum1, 51, 3)   # 
    #xSum1 = np.gradient(xSum1) #
    #xSum1 = savgol_filter(xSum1, 51, 3) #
    #Sum1 = np.gradient(np.gradient(xSum1)) #
    #xSum1 = savgol_filter(xSum1, 51, 3) #
    
    #s = int(15*ii/32)
    #e = int(17*ii/32)
    #cgx  = np.argmin(xSum1[s:e])+s
    #cgx  = np.argmax(xSum1[125:250])+125
    #cgx = (xSum1*np.arange(len(xSum1))).sum()/xSum1.sum()
    cgx  = np.argmin(xSum1)
    #s = np.argmin(xSum1[0:150])
    #s = 150
    #e = len(xSum1)-50
    #e = np.argmin(xSum1)
    #cgx  = np.argmax(xSum1[s:e])+s
    return cgx, xSum1 
                                
# auxiliary functions:         
#---------------------------------------------------------------------------    
def transform_stack_to_stack_of_square_images(stack):
    s = stack.shape
    L = int(np.sqrt(s[1]*s[2]))    
    out = np.zeros((s[0],L,L))
    for k in range(0,s[0]):
        out[k,:,:] = np.squeeze(stack[k,:,:]).reshape((L,L))
    return out            
#---------------------------------------------------------------------------
def timestamp():    
    ct = datetime.datetime.now().isoformat()
    s0 = ct.split('T')                
    s = s0[1].split(':')
    return s0[0] + '-' + s[0] + '-' + s[1]                
#---------------------------------------------------------------------------
def createFolder(path,folder):
    # 
    directory = path + os.path.sep + folder + os.path.sep
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    #
    assert(os.path.isdir(directory))
    # 
    return directory
#---------------------------------------------------------------------------
def normalize_by_percentile(img,pmin,pmax):
    x = img.copy()
    xmin = np.percentile(x,pmin)
    xmax = np.percentile(x,pmax)
    return (x - xmin)/(xmax - xmin)  
#---------------------------------------------------------------------------
def getImageInfo(filename):
    md = bf.get_omexml_metadata(filename)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    pT = pixels.SizeT
    pX = pixels.SizeX
    pY = pixels.SizeY
    return pT, pX, pY
#---------------------------------------------------------------------------
def save_model_summary(path, model):
    with open(path + 'model_summary.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    tf.keras.utils.plot_model(model, to_file= path+"model_plot.png", show_shapes=True, show_layer_names=True)

# plotting functions
#---------------------------------------------------------------------------                    
def accCalc(listPred, listY, n_frames, results_folder, token, zStep):        
    l = len(listPred)
    ll = len(listPred[0])
    binn = zStep
    meanArr = np.zeros((int(n_frames*zStep/binn), ll*l))
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 1.5, 1])
    for i in range(0, l):
        lisP = listPred[i]
        lisY = listY[i]
        #print(len(lisY))
        #print(len(lisP))
        for ii in range(0, ll):
            
            #print(i*ll+ii)
            #print(lisP[ii])
            num = int((lisY[ii]+n_frames*zStep/2)/binn)
            if num>=ll:
                num = ll-1
            if num<0:
                num=0
            meanArr[num, i*ll+ii] = lisP[ii]
    mm11 = np.where(np.isclose(meanArr,0), np.nan, meanArr)
    mm1 = np.nanstd(mm11, axis=1)
    ax2.plot(lisY, mm1)   
    plt.title('Consistency', fontsize = 20)
    plt.xlabel(' stage value [\u03BCm]', fontsize = 18)
    plt.ylabel('std [\u03BCm]', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim((0,5))
    #plt.ylim(0,graph_y_lim)
    pp = results_folder + '\\' + token +'_norm_acc.png'
    fig2.savefig(pp, format='png', bbox_inches='tight')    
    
    plt.show(block=False)
    time.sleep(5)
    plt.close('all') 
#---------------------------------------------------------------------------           
def residMean(listResid, listY, n_frames, results_folder, token, zStep):
    #l = len(listResid)
    ll = len(listY)
    new_listY = []
    for i in range(ll):
        lll = len(listY[i])
        for j in range(lll):
            if listY[i][j] not in new_listY:
                new_listY.append(listY[i][j])
    new_listY = sorted(new_listY)
    mean_list = []
    std_list = []
    abs_std_list = []
    abs_mean_list = []
    max_list = []
    min_list = []
    upper_quartile_list = []
    lower_quartile_list = []
    for i in new_listY:
        sum_list = []
        abs_sum_list = []
        for j in range(ll):
            if i in listY[j]:
                masked_lY = listY[j] == i
                masked_lY = masked_lY.astype(np.int)
                masked_listRes = masked_lY * listResid[j]
                sum_list.append(np.sum(masked_listRes))
                abs_sum_list.append(np.abs(np.sum(masked_listRes)))
                
        max_ = np.max(sum_list)
        min_ = np.min(sum_list) 
        quartile_75 = np.percentile(sum_list, 25)
        quartile_25 = np.percentile(sum_list, 75)
        mean_ = np.mean(sum_list)
        std_ = np.std(sum_list)
        abs_std_ = np.std(abs_sum_list)
        abs_mean_ = np.mean(abs_sum_list)
        std_list.append(std_)
        mean_list.append(mean_)
        abs_std_list.append(abs_std_)
        abs_mean_list.append(abs_mean_)
        max_list.append(max_)
        min_list.append(min_)
        upper_quartile_list.append(quartile_75)
        lower_quartile_list.append(quartile_25)
    
    fig4 = plt.figure()
    ax4 = fig4.add_axes([0.1, 0.1, 1.5, 1])
    ax4.plot(new_listY, mean_list, zorder = 10)   
    #print(np.shape(new_listY))
    #print(np.shape(mean_list))
    #print(np.shape(std_list))
    plt.fill_between(new_listY, np.asarray(mean_list)-np.asarray(std_list), np.asarray(mean_list)+np.asarray(std_list), color='gray', alpha=0.5, zorder = 5)
    #plt.fill_between(new_listY, min_list, max_list, color='gray', alpha=0.5)
    #plt.plot(new_listY, upper_quartile_list, color='k', zorder = 5, alpha=0.5)
    #plt.plot(new_listY, lower_quartile_list, color='k', zorder = 6, alpha=0.5)
    
    #plt.fill_between(new_listY, lower_quartile_list, upper_quartile_list, alpha=0.5, color='orange', zorder = 4)
    #plt.fill_between(new_listY, lower_quartile_list, upper_quartile_list, alpha=0.5, color='gray', zorder = 5)
    #plt.errorbar(x=new_listY, y=mean_list, yerr = [np.asarray(mean_list)-np.asarray(min_list),np.asarray(max_list)-np.asarray(mean_list)], alpha=0.5, color = "k",zorder = 0)
    #plt.fill_between(new_listY, min_list, max_list, alpha=0.2, color = "gray",zorder = 0)
    rect = plt.Rectangle((-100,-0.3),200,0.6, edgecolor='r', facecolor="r", alpha = 0.1)
    ax4.add_patch(rect)
    #ax4.errorbar(x=new_listY, y=mean_list, yerr = std_list)
    plt.title('Residual error', fontsize = 20)
    plt.xlabel('Defocus value [\u03BCm]', fontsize = 18)
    plt.ylabel('Predicted value - Defocus value [\u03BCm]', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim(-5,5)
    #plt.xlim(-10,10)
    #plt.ylim(-graph_y_lim,graph_y_lim)
    pp = results_folder + '\\' + token + '_residual_mean.png'
    custom_lines = [Line2D([0], [0], color='b', lw=4),
        Line2D([0], [0], color='gray', lw=4, alpha = 0.5),
        Line2D([0], [0], color='red', lw=4, alpha = 0.1)]
    #ax4.legend(custom_lines, ['Mean Residual', 'Interquartile range', 'Max and Min error range'],prop={"size":14}, loc='upper right')
    ax4.legend(custom_lines, ['Mean over 10 repeats', 'Standard Deviation', 'Confocal Parameter Range'],prop={"size":14}, loc='upper right').set_zorder(15)
    fig4.savefig(pp, format='png', bbox_inches='tight')
    
    fig9 = plt.figure()
    ax9 = fig9.add_axes([0.1, 0.1, 1.5, 1])
    ax9.plot(new_listY, abs_mean_list)     
    #ax9.errorbar(x=new_listY, y=mean_list, yerr = std_list)
    plt.title('Absolute residual mean', fontsize = 20)
    plt.xlabel('Stage value [\u03BCm]', fontsize = 18)
    plt.ylabel('STD of absolute residual [\u03BCm]', fontsize = 18)  
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim(0,5)
    #plt.ylim(0,graph_y_lim)
    pp = results_folder + '\\' + token + '_abs_residual_mean.png'    
    fig9.savefig(pp, format='png', bbox_inches='tight')
    
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')            
    
    return mean_list
#---------------------------------------------------------------------------
def make_plots(listPred, listY, listResid, n_frames, results_folder, token, zStep):

    plt.show(block=False)
    time.sleep(5)
    plt.close('all')    

    half_range = (n_frames-1)*zStep/2.0
                    
    fig = plt.figure(1)
    ax = fig.add_axes([0.1, 0.1, 1.5, 1])
    plt.title('Validation stacks', fontsize = 20)    
    plt.xlabel('Defocus value [\u03BCm]', fontsize = 18)
    plt.ylabel('Predicted value [\u03BCm]', fontsize = 18)
    #plt.xlim(-1.4*half_range,1.4*half_range)
    #plt.ylim(-1.4*half_range,1.4*half_range)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig3 = plt.figure(2)
    ax3 = fig3.add_axes([0.1, 0.1, 1.5, 1])
    plt.ylim(-half_range/8,half_range/8)
         
    for k in range(0, len(listY)):
        xVal = listY[k] 
        yVal = listPred[k]                          
        resid = listResid[k]        
        ax.plot(xVal, yVal , label=str(k))
        ax3.plot(xVal, resid , label=str(k))  
                                    
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. ,prop={"size":14})
    plt.title('validation Residual', fontsize = 20)     
    plt.xlabel('Stage value [\u03BCm]', fontsize = 18)
    plt.ylabel('Predicted value - Stage value [\u03BCm]', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #
    pp = results_folder + '\\' + token + '_defocus_gt_vs_predicted' + '.png'
    fig.savefig(pp, format='png', bbox_inches='tight')
    pp = results_folder + '\\' + token + '_residuals_vs_stage_value.png'
    fig3.savefig(pp, format='png', bbox_inches='tight')

    plt.show(block=False)
    time.sleep(5)
    plt.close('all') 
