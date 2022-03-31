import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten, BatchNormalization,Activation,Dense
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from tensorflow.keras import regularizers
from os.path import isfile, join
import os
import random
from tensorflow.keras.callbacks import ModelCheckpoint
#%%
def predict(xArr, xFFT, yArr, modelLoad, load_path, zStep):
    xA = xArr
    xFF = xFFT
    yA = yArr
    l=len(xA)

    if True:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1])
        for i in range(0, 99):
            numSlice = 251
            x = xA[i*numSlice:(i+1)*numSlice,:,:]
            xF = xFF[i*numSlice:(i+1)*numSlice,:,:]
            y = yA[i*numSlice:(i+1)*numSlice]
            s = x.shape
            x_val = np.zeros([s[0], s[1], s[2], 2])
            x_val[:,:,:,0] = x
            x_val[:,:,:,1] = xF
            pred = modelLoad.predict(x_val)
            diff = np.subtract(pred,y)
            met = np.sqrt(np.sum(diff**2)/numSlice)
            lab = str(i) + ": metric: " + str(met)
            ax.plot(y, pred, label=lab)
            plt.title('first 251')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

       
        numSlice = 251
        i = 99
        xA1 = []
        xA1 =xA[numSlice*i:l ,:,:]
        xFF1 = []
        xFF1 =xFF[numSlice*i:l ,:,:]
        yA1 = []
        yA1 =yA[numSlice*i:l]
    
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1])
        for i in range(0, 153):
            numSlice = 101
            x = xA1[i*numSlice:(i+1)*numSlice,:,:]
            xF = xFF1[i*numSlice:(i+1)*numSlice,:,:]
            y = yA1[i*numSlice:(i+1)*numSlice]
            s = x.shape
            x_val = np.zeros([s[0], s[1], s[2], 2])
            x_val[:,:,:,0] = x
            x_val[:,:,:,1] = xF
            pred = modelLoad.predict(x_val)
            diff = np.subtract(pred,y)
            met = np.sqrt(np.sum(diff**2)/numSlice)
            lab = str(i) + ": metric: " + str(met)
            ax.plot(y, pred, label=lab)
            plt.title('first 101')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


        xA2 = []
        xA2 =xA1[numSlice*i:l ,:,:]
        xFF2 = []
        xFF2 =xFF1[numSlice*i:l ,:,:]
        yA2 = []
        yA2 =yA1[numSlice*i:l]
    
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1])
        for i in range(0, 26):
            numSlice = 251
            x = xA2[i*numSlice:(i+1)*numSlice,:,:]
            xF = xFF2[i*numSlice:(i+1)*numSlice,:,:]
            y = yA2[i*numSlice:(i+1)*numSlice]
            s = x.shape
            x_val = np.zeros([s[0], s[1], s[2], 2])
            x_val[:,:,:,0] = x
            x_val[:,:,:,1] = xF
            pred = modelLoad.predict(x_val)
            diff = np.subtract(pred,y)
            met = np.sqrt(np.sum(diff**2)/numSlice)
            lab = str(i) + ": metric: " + str(met)
            ax.plot(y, pred, label=lab)
            plt.title('second 251')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    if True: 
        xA3 = []
        xA3 =xA2[numSlice*i:l ,:,:]
        xFF3 = []
        xFF3 =xFF2[numSlice*i:l ,:,:]
        yA3 = []
        yA3 =yA2[numSlice*i:l]
    
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1])
        for i in range(0, 45):
            numSlice = 101
            x = xA3[i*numSlice:(i+1)*numSlice,:,:]
            xF = xFF3[i*numSlice:(i+1)*numSlice,:,:]
            y = yA3[i*numSlice:(i+1)*numSlice]
            s = x.shape
            x_val = np.zeros([s[0], s[1], s[2], 2])
            x_val[:,:,:,0] = x
            x_val[:,:,:,1] = xF
            pred = modelLoad.predict(x_val)
            diff = np.subtract(pred,y)
            met = np.sqrt(np.sum(diff**2)/numSlice)
            lab = str(i) + ": metric: " + str(met)
            ax.plot(y, pred, label=lab)
            plt.title('second 101')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        xA4 = []
        xA4 =xA3[numSlice*i:l+1 ,:,:]
        xFF4 = []
        xFF4 =xFF3[numSlice*i:l+1,:,:]
        yA4 = []
        yA4 =yA3[numSlice*i:l+1]
        
        print(len(xA4))
        print(len(xFF4))
        print(len(yA4))
    return pred

#%%
def createFolders(path):
    data_path = path + "cnn_data\\"
    res_path = path + "cnn_results\\"  
    directory = os.path.dirname(data_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
        
    directory = os.path.dirname(res_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
    return data_path, res_path
#%%
def createFolders_Proj(path):
    data_path = path + "cnn_data_Proj\\"
    res_path = path + "cnn_results_Proj\\"  
    directory = os.path.dirname(data_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
        
    directory = os.path.dirname(res_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
    return data_path, res_path

#%%
def comb_loadMaster(path):
    pp =path + "x_train.npy"
    xArr = np.load(pp)
    pp =path + "fft_x_train.npy"
    xFFT = np.load(pp)
    pp = path + "y_train.npy"
    yArr = np.load(pp)
    #pp =  path + "com_x_train.npy"
    #com = np.load(pp)
    return xArr, yArr, xFFT

#%%
def comb_loadMaster_Proj(path):
    pp =path + "x_proj_train.npy"
    xArr = np.load(pp)
    #pp =path + "fft_x_train.npy"
    #xFFT = np.load(pp)
    pp = path + "y_proj_train.npy"
    yArr = np.load(pp)
    #pp =  path + "com_x_train.npy"
    #com = np.load(pp)
    return xArr, yArr#, xFFT
#%%
def comb_trainModel(xArr, yArr, xFFT, model, res_path, epos):
    
    s = xArr.shape

    x_train = np.zeros([s[0], s[1], s[2], 2])
    #train
    x_train[:,:,:,0] = xArr
    x_train[:,:,:,1] = xFFT

    history =model.fit(x_train, # Features
                      yArr, # Target vector
                      epochs=epos,#, # Number of epochs
                      validation_split=0.5,
                      verbose=1, # Print description after each epoch
                      batch_size=20, # Number of observations per batch
                      )

  
    #loss = history.history['loss']
    acc = history.history['mean_squared_error']
    acc_val = history.history['val_mean_squared_error']
    # save model
    pp = res_path + "model"
    model.save(pp)
    pp = res_path + "acc.npy"
    np.save(pp, acc)
    pp = res_path + "acc_val.npy"
    np.save(pp, acc_val)
    
#%% 
def saveWeights(model, res_path):
    model.summary()
    l1 = model.layers[0].get_weights()[0]
    b1 =  model.layers[0].get_weights()[1]
    l2 = model.layers[3].get_weights()[0]
    b2 =  model.layers[3].get_weights()[1]
    l3 = model.layers[8].get_weights()[0]
    b3 =  model.layers[8].get_weights()[1]
    l4 = model.layers[9].get_weights()[0]
    b4 =  model.layers[9].get_weights()[1]
    
    pp = res_path + "weights_l0.npy"
    np.save(pp, l1)
    pp = res_path + "bias_l0.npy"
    np.save(pp, b1)
    pp = res_path + "weights_l1.npy"
    np.save(pp, l2)
    pp = res_path + "bias_l1.npy"
    np.save(pp, b2)
    pp = res_path + "weights_l4.npy"
    np.save(pp, l3)
    pp = res_path + "bias_l4.npy"
    np.save(pp, b3)
    pp = res_path + "weights_l5.npy"
    np.save(pp, l4)
    pp = res_path + "bias_l5.npy"
    np.save(pp, b4)
#%%
def weights_table(path):
    weights_abs_mean = []
    weights_abs_max = []
    layers = []
    for file in os.listdir(path):
        if 'weights_' in file:
            if '.npy' in file:
                we = np.load(path+file)
                abs_mean = np.mean(np.abs(we))
                abs_max = np.max(np.abs(we))
                weights_abs_mean.append(abs_mean)
                weights_abs_max.append(abs_max)
                name = file.split('_')
                layer = name[1].split('.npy')[0]
                layers.append(layer)
    data = [weights_abs_mean, weights_abs_max]
    rows = ['Abs Mean','Abs Max']
    print(weights_abs_mean)
    print(weights_abs_max)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    tb_mean=ax.table(cellText=np.around(data,4), rowLabels=rows, colLabels=layers, loc='center')
    plt.gcf().subplots_adjust()
    #plt.show()
    plt.savefig(path+"\\weights.png", bbox_inches="tight")
   
#%%
def comb_trainModelWithSeperateDayValid(xArr, yArr, xFFT, model, res_path, epos, x_valid, y_valid, num):
    
    s = xArr.shape

    #x_train = np.zeros([s[0], s[1], s[2], 2]) # normal amnd fourier space
    x_train = np.zeros([s[0], s[1], s[2], 2]) # just one space
    #train
    x_train[:,:,:,0] = xArr
    x_train[:,:,:,1] = xFFT

    xN = np.zeros([num, s[1], s[2], 2])
    yN = np.zeros([num])
    
# =============================================================================
#     l = []
#     for i in range(0,num):
#         c = np.random.randint(0,s[0],1)
#         while c in l:
#             c = np.random.randint(0,s[0],1)
#             if c not in l:
#                 break
#         l.append(c)
#         xN[i,:,:,:] = x_train[c,:,:,:]
#         yN[i] = yArr[c]
# 
#     dupes = [x for n, x in enumerate(l) if x in l[:n]]
#     print(dupes)  
# =============================================================================

    rand_l = random.sample(range(0, s[0]), num)
    dupes = [x for n, x in enumerate(rand_l) if x in rand_l[:n]]
    print('dupes',dupes) 
    for i in range(len(rand_l)):
        xN[i,:,:,:] = x_train[rand_l[i],:,:,:]
        yN[i] = yArr[rand_l[i]]
    
    #best epoch callback    
# =============================================================================
#     filepath=res_path+"weights_best.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
#     callbacks_list = [checkpoint]
# =============================================================================
    
    #earlyStopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history =model.fit(xN, # Features
                      yN, # Target vector
                      epochs=epos,#, # Number of epochs
                      validation_data=(x_valid, y_valid),
                      verbose=1, # Print description after each epoch
                      batch_size=128, # Number of observations per batch
                      #callbacks=callbacks_list # callbacks best model
                      callbacks = [callback]
                      )

     #loss = history.history['loss']
    acc = history.history['mean_squared_error']
    acc_val = history.history['val_mean_squared_error']
    # save model
    pp = res_path + "model"
    model.save(pp)
    pp = res_path + "acc.npy"
    np.save(pp, acc)
    pp = res_path + "acc_val.npy"
    np.save(pp, acc_val)
    
    
# =============================================================================
#     print(history.history.keys())
#     plt.plot(history.history['mean_absolute_percentage_error'])
#     plt.plot(history.history['val_mean_absolute_percentage_error'])
#     plt.title('mean_absolute_percentage_error vs epoch')
#     plt.ylabel('mean_absolute_percentage_error')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# =============================================================================
    
    
    return acc, acc_val   
    
            
#%%
def comb_trainModelWithValid(xArr, yArr, xFFT, model, res_path, epos, x_valid, y_valid, num):
    
    s = xArr.shape

    #x_train = np.zeros([s[0], s[1], s[2], 2]) # normal amnd fourier space
    x_train = np.zeros([s[0], s[1], s[2], 2]) # just one space
    #train
    x_train[:,:,:,0] = xArr
    x_train[:,:,:,1] = xFFT

    xN = np.zeros([num, s[1], s[2], 2])
    yN = np.zeros([num])
    
# =============================================================================
#     l = []
#     for i in range(0,num):
#         c = np.random.randint(0,s[0],1)
#         while c in l:
#             c = np.random.randint(0,s[0],1)
#             if c not in l:
#                 break
#         l.append(c)
#         xN[i,:,:,:] = x_train[c,:,:,:]
#         yN[i] = yArr[c]
# 
#     dupes = [x for n, x in enumerate(l) if x in l[:n]]
#     print(dupes)  
# =============================================================================

    rand_l = random.sample(range(0, s[0]), num)
    dupes = [x for n, x in enumerate(rand_l) if x in rand_l[:n]]
    print('dupes',dupes) 
    for i in range(len(rand_l)):
        xN[i,:,:,:] = x_train[rand_l[i],:,:,:]
        yN[i] = yArr[rand_l[i]]
    
    #best epoch callback    
# =============================================================================
#     filepath=res_path+"weights_best.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
#     callbacks_list = [checkpoint]
# =============================================================================
    
    #earlyStopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history =model.fit(xN, # Features
                      yN, # Target vector
                      epochs=epos,#, # Number of epochs
                      validation_data=(x_valid, y_valid),
                      verbose=1, # Print description after each epoch
                      batch_size=128, # Number of observations per batch
                      #callbacks=callbacks_list # callbacks best model
                      callbacks = [callback]
                      )

     #loss = history.history['loss']
    acc = history.history['mean_squared_error']
    acc_val = history.history['val_mean_squared_error']
    # save model
    pp = res_path + "model"
    model.save(pp)
    pp = res_path + "acc.npy"
    np.save(pp, acc)
    pp = res_path + "acc_val.npy"
    np.save(pp, acc_val)
    
    
# =============================================================================
#     print(history.history.keys())
#     plt.plot(history.history['mean_absolute_percentage_error'])
#     plt.plot(history.history['val_mean_absolute_percentage_error'])
#     plt.title('mean_absolute_percentage_error vs epoch')
#     plt.ylabel('mean_absolute_percentage_error')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# =============================================================================
    
    
    return acc, acc_val
#%%
def comb_trainModelWithValid_MobileNetV2(xArr, yArr, xFFT, model, res_path, epos, x_valid, y_valid, num):
    
    s = xArr.shape

    #x_train = np.zeros([s[0], s[1], s[2], 2]) # normal amnd fourier space
    x_train = np.zeros([s[0], s[1], s[2], 1]) # just one space
    #train
    x_train[:,:,:,0] = xArr
    #x_train[:,:,:,1] = xFFT

    xN = np.zeros([num, s[1], s[2], 1])
    yN = np.zeros([num])
    
# =============================================================================
#     l = []
#     for i in range(0,num):
#         c = np.random.randint(0,s[0],1)
#         while c in l:
#             c = np.random.randint(0,s[0],1)
#             if c not in l:
#                 break
#         l.append(c)
#         xN[i,:,:,:] = x_train[c,:,:,:]
#         yN[i] = yArr[c]
# 
#     dupes = [x for n, x in enumerate(l) if x in l[:n]]
#     print(dupes)  
# =============================================================================

    rand_l = random.sample(range(0, s[0]), num)
    #dupes = [x for n, x in enumerate(rand_l) if x in rand_l[:n]]
    #print('dupes',dupes) 
    for i in range(len(rand_l)):
        xN[i,:,:,:] = x_train[rand_l[i],:,:,:]
        yN[i] = yArr[rand_l[i]]
    
    #best epoch callback    
    filepath=res_path+"weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    x_train_tensor = tf.convert_to_tensor(xN)
    x_valid_tensor = tf.convert_to_tensor(x_valid)
    #earlyStopping
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    batchSize = 128
    stepsPerEpoch = int(s[0]/batchSize)
    
    history =model.fit(x_train_tensor, # Features
                      yN, # Target vector
                      epochs=epos,#, # Number of epochs
                      validation_data=(x_valid_tensor, y_valid),
                      steps_per_epoch = stepsPerEpoch, 
                      verbose=1, # Print description after each epoch
                      #batch_size=128, # Number of observations per batch
                      callbacks=callbacks_list # callbacks best model
                      #callbacks = [callback]
                      )
    model.build((96,96,1))
    #loss = history.history['loss']
    acc = history.history['mse']
    acc_val = history.history['val_mse']
    # save model
    pp = res_path + "model"
    model.save(pp)
    #tf.keras.models.save_model(model, pp)
    pp = res_path + "acc.npy"
    np.save(pp, acc)
    pp = res_path + "acc_val.npy"
    np.save(pp, acc_val)
    
    
# =============================================================================
#     print(history.history.keys())
#     plt.plot(history.history['mean_absolute_percentage_error'])
#     plt.plot(history.history['val_mean_absolute_percentage_error'])
#     plt.title('mean_absolute_percentage_error vs epoch')
#     plt.ylabel('mean_absolute_percentage_error')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# =============================================================================
    
    
    return acc, acc_val
#%%
def plotModelRes(res_path):
    pp = res_path + "acc.npy"
    acc = np.load(pp)
    pp = res_path + "acc_val.npy"
    acc_val = np.load(pp)
    fig3 = plt.figure()
    ax3 = fig3.add_axes([0.1, 0.1, 1.5, 1])
    ax3.plot(acc, label="loss per epoch")
    ax3.plot(acc_val, label="validation loss")
    plt.title('epoch and validation')
    plt.xlabel(' epochs ')
    plt.ylabel('loss')
    #plt.ylim((0,1))
    plt.yscale("log")
    pp = res_path + 'lossVSepoch2.png'
    fig3.savefig(pp, format='png', bbox_inches='tight')
    
#%%
def comb_validate(load_path, save_path, valid_path, count):
    pp = load_path + "model"
    modelLoad = tf.keras.models.load_model(pp)
    direc = [f for f in listdir(valid_path) if isfile(join(valid_path, f))]
    ll= len(direc)
    for ii in range(0, ll):
        if "x_val_stack" in direc[ii]:
            print(direc[ii])
            pp = valid_path + direc[ii]
            x_val = np.load(pp)
            pp1 = pp.replace('x_val_','y_val_')
            y_val = np.load(pp1)
            y_val = y_val
            pp1 = pp.replace('x_val_','x_val_fft_')
            xFFT = np.load(pp1)
            s = x_val.shape
            x_val1 = np.zeros([s[0], s[1], s[2], 2])
            x_val1[:,:,:,0] = x_val
            x_val1[:,:,:,1] = xFFT
            predictions = modelLoad.predict(x_val1)
            st = 'x_pred_' + str(count) + '_'
            pp = save_path + direc[ii].replace('x_val_',st)
            np.save(pp, predictions)
            st = 'y_val_' + str(count) + '_'
            pp = save_path + direc[ii].replace('x_val_', st)
            np.save(pp, y_val)
            
#%%            
def comb_plotValid(valid_path, res_path, zStep):
    #pp =valid_path + "numSlice.npy"
    #numSlice = np.load(pp)
    files = [f for f in listdir(valid_path) if isfile(join(valid_path, f))]
    ll= len(files)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 1.5, 1])
    listPred = []
    listY = []
    for ii in range(0, ll):
        if "y_val_" in files[ii]:
            yName = files[ii]
            stack = yName.replace('y_val_','')
            predName = yName.replace('y_val_','x_pred_')
            pp =valid_path + yName
            yVal = np.load(pp)
            yVal = yVal
            pp =valid_path + predName
            predVal = np.load(pp)
            ax.plot(yVal, predVal, label=stack)  
            listPred.append(predVal)
            listY.append(yVal)
            
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('validation stacks')    
    plt.xlabel(' stage value [um]')
    plt.ylabel('predicted value [um]')
    pp = res_path + 'add_validation.png'
    fig.savefig(pp, format='png', bbox_inches='tight')
#%%
def combArrLoop(pList):
    l= len(pList)
    s = []
    for i in range(1, l):
        path = pList[i]
        pp = path + "cnn_data\\" + "y_train.npy"
        yArr1 = np.load(pp)
        s.append(yArr1.shape)
        
    ss = 0
    for i in range(0, l-1):
        sh = s[i]
        ss = ss + sh[0]
        
    path = pList[1]
    pp =path + "cnn_data\\" + "x_train.npy"
    xArr1 = np.load(pp)
    pp =path + "cnn_data\\" + "x_fft_train.npy"
    xFFT1 = np.load(pp)
    pp = path + "cnn_data\\" + "y_train.npy"
    yArr1 = np.load(pp)
    
    s1 = xArr1.shape
    newX = np.ones((ss, s1[1], s1[2]))
    newFFT = np.ones((ss, s1[1], s1[2]))
    newY = np.ones((ss))
    newX[0:s1[0], :, :] = xArr1
    newFFT[0:s1[0], :, :] = xFFT1
    newY[0:s1[0]] = yArr1
    
    for i in range(0, l-2):
        path = pList[i+2]
        print(path)
        pp =path + "cnn_data\\" + "x_train.npy"
        xArr2 = np.load(pp)
        pp =path + "cnn_data\\" + "x_fft_train.npy"
        xFFT2 = np.load(pp)
        pp = path + "cnn_data\\" + "y_train.npy"
        yArr2 = np.load(pp)
        sSt = 0
        for ii in range(0, i+1):
            shh = s[ii]
            sSt = sSt + shh[0]
        shh = s[i+1]
        sEn = sSt + shh[0]
        print(s)
        print(xArr2.shape)
        print(shh)
        print(sSt)
        print(sEn)
        newX[sSt:sEn, :, :] = xArr2
        newFFT[sSt:sEn, :, :] = xFFT2
        newY[sSt:sEn] = yArr2
    
    path = pList[0]
    pp = path + "cnn_data\\" + "x_train.npy"
    np.save(pp, newX)
    pp = path + "cnn_data\\" + "fft_x_train.npy"
    np.save(pp, newFFT)
    pp = path + "cnn_data\\" + "y_train.npy"
    np.save(pp, newY)    
#%%
def combArrLoop_Proj(pList):
    l= len(pList)
    s = []
    for i in range(1, l):
        path = pList[i]
        pp = path + "cnn_data_Proj\\" + "y_proj_train.npy"
        yArr1 = np.load(pp)
        s.append(yArr1.shape)
        
    ss = 0
    for i in range(0, l-1):
        sh = s[i]
        ss = ss + sh[0]
        
    path = pList[1]
    pp =path + "cnn_data_Proj\\" + "x_proj_train.npy"
    xArr1 = np.load(pp)
    #pp =path + "cnn_data_Proj\\" + "x_fft_train.npy"
    #xFFT1 = np.load(pp)
    pp = path + "cnn_data_Proj\\" + "y_proj_train.npy"
    yArr1 = np.load(pp)
    
    s1 = xArr1.shape
    newX = np.ones((ss, s1[1], s1[2]))
    #newFFT = np.ones((ss, s1[1], s1[2]))
    newY = np.ones((ss))
    newX[0:s1[0], :, :] = xArr1
    #newFFT[0:s1[0], :, :] = xFFT1
    newY[0:s1[0]] = yArr1
    
    for i in range(0, l-2):
        path = pList[i+2]
        print(path)
        pp =path + "cnn_data_Proj\\" + "x_proj_train.npy"
        xArr2 = np.load(pp)
        #pp =path + "cnn_data_Proj\\" + "x_fft_train.npy"
        #xFFT2 = np.load(pp)
        pp = path + "cnn_data_Proj\\" + "y_proj_train.npy"
        yArr2 = np.load(pp)
        sSt = 0
        for ii in range(0, i+1):
            shh = s[ii]
            sSt = sSt + shh[0]
        shh = s[i+1]
        sEn = sSt + shh[0]
        print(s)
        print(xArr2.shape)
        print(shh)
        print(sSt)
        print(sEn)
        newX[sSt:sEn, :, :] = xArr2
        #newFFT[sSt:sEn, :, :] = xFFT2
        newY[sSt:sEn] = yArr2
    
    path = pList[0]
    pp = path + "cnn_data_Proj\\" + "x_proj_train.npy"
    np.save(pp, newX)
    #pp = path + "cnn_data\\" + "fft_x_train.npy"
    #np.save(pp, newFFT)
    pp = path + "cnn_data_Proj\\" + "y_proj_train.npy"
    np.save(pp, newY)    

#%%
    
    
    
def combArr(pList):
    path = pList[1]
    pp =path + "analysed_data\\" + "x_train.npy"
    xArr1 = np.load(pp)
    pp =path + "analysed_data\\" + "fft_x_train.npy"
    xFFT1 = np.load(pp)
    pp = path + "analysed_data\\" + "y_train.npy"
    yArr1 = np.load(pp)
    
    path = pList[2]
    pp =path + "analysed_data\\" + "x_train.npy"
    xArr2 = np.load(pp)
    pp =path + "analysed_data\\" + "fft_x_train.npy"
    xFFT2 = np.load(pp)
    pp = path + "analysed_data\\" + "y_train.npy"
    yArr2 = np.load(pp)
    
    s1 = xArr1.shape
    s2 = xArr2.shape
    ss = s1[0]+s2[0]

    newX = np.ones((ss, s1[1], s2[2]))
    newX[0:s1[0], :, :] = xArr1
    newX[s1[0]:ss, :, :] = xArr2
    
    newFFT = np.ones((ss, s1[1], s2[2]))
    newFFT[0:s1[0], :, :] = xFFT1
    newFFT[s1[0]:ss, :, :] = xFFT2
    
    newY = np.ones((ss))
    newY[0:s1[0]] = yArr1
    newY[s1[0]:ss] = yArr2
    
    path = pList[0]
    pp = path + "analysed_data\\" + "x_train.npy"
    np.save(pp, newX)
    pp = path + "analysed_data\\" + "fft_x_train.npy"
    np.save(pp, newFFT)
    pp = path + "analysed_data\\" + "y_train.npy"
    np.save(pp, newY)
    
#%%
def comb_genY_train(path, zStep):
    pp =path + "x_train.npy"
    xArr = np.load(pp)
    pp =path + "numSlice.npy"
    numSlice = np.load(pp)
    s = xArr.shape
    xSum1 = []
    int(s[0]/numSlice)
    y = []
    
    for i in range(0, int(s[0]/numSlice)):
        x = xArr[i*numSlice:numSlice*(i+1)]
        if numSlice*zStep > 75:
            cgx, xSum1 = center(x, xSum1, numSlice, i) 
        else:
            cgx, xSum1 = centerShort(x, xSum1, numSlice, i)
        yS = 0-int(cgx)
        yE = numSlice-int(cgx)
        y1 = np.arange(yS, yE, 1)*zStep
        y[i*numSlice:(i+1)*numSlice] = y1

        plt.pause(0.05)
        plt.plot( y1,xSum1) 
        plt.title(i)   
        xSum1 = []
    pp = path + "y_train"
    np.save(pp, y)
    
    
#%%
def defModel(xArr):
    s = xArr.shape
    '''
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32,(3,3), input_shape = (s[1], s[2], 2), activation=tf.nn.relu))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1)) 
    '''
    '''
    #new Model
    model = tf.keras.models.Sequential()
    model.add(Conv2D(2,(3,3), input_shape = (s[1], s[2], 2),kernel_regularizer = regularizers.l2(0.01)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(2,(5,5),kernel_regularizer = regularizers.l2(0.01)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (8,8)))
    model.add(Flatten())
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))
    '''
    model = tf.keras.models.Sequential()
    model.add(Conv2D(2,(3,3), input_shape = (s[1], s[2], 2),kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(2,(5,5),kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(2,(7,7),kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(2,(9,9),kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))
    

    adam2 = tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(optimizer = adam2,
                  #loss = 'sparse_categorical_crossentropy',
                  loss = 'mean_squared_error',
                  metrics = ['mse'])
    #train
    
    return model
#%%
def defModel_MobileNetV2(xArr,image_shape):

    base_model=tf.keras.applications.MobileNetV2(weights=None,include_top=False,input_shape=(image_shape,image_shape,1)) #imports the MobileNetV2 model and discards the last 1000 neuron layer.
    x=base_model.output
    x = GlobalAveragePooling2D()(x)
    preds=Dense(1)(x)
    model=tf.keras.Model(inputs=base_model.input,outputs=preds)
    

    adam2 = tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(optimizer = adam2,
                  #loss = 'sparse_categorical_crossentropy',
                  loss = 'mse',
                  metrics = ['mse'])
    #train
    
    return model

#%%
def transferLearn_Model_MobileNetV2(pp, fine_tune_at):

    pp_model = pp + "model/"
    base_model = tf.keras.models.load_model(pp_model)
    filepath=pp+"weights_best.hdf5"
    base_model.load_weights(filepath)

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    adam2 = tf.keras.optimizers.Adam(lr = 0.00001)
    base_model.compile(optimizer = adam2,
                  #loss = 'sparse_categorical_crossentropy',
                  loss = 'mse',
                  metrics = ['mse'])
    #train
    
    return base_model
#%%
def save_model_summary(path, model):
    with open(path + 'model_summary.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    tf.keras.utils.plot_model(model, to_file= path+"model_plot.png", show_shapes=True, show_layer_names=True)

#%%
# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

