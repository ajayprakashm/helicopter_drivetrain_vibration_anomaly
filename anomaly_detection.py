# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:49:10 2019

@author: Ajay
"""
#%% Importing fuction 
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
import scipy.io as spio
from scipy import stats
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.io as spio
warnings.filterwarnings("ignore")
plt.close('all')
sns.set(style="whitegrid")
np.random.seed(203)
# modify the path to correspond to where your data is
os.chdir('D:\Assignment')
mat1 = spio.loadmat('IGBOutput7.mat')
mat2 = spio.loadmat('IGBOutput21.mat')
mat3 = spio.loadmat('IGBOutput34.mat')

d1 = mat1['Output'][0]      # "7" is "normal"
d2 = mat2['Output'][0]      # "21" is "mildly faulted"
d3 = mat3['Output'][0]      # "34" is "very faulted"

#%% Functions used in this code
# This function will chunk a vector up into blocks of length cwidth, advancing the 
# window every stepsize samples. Returns a matrix with one block per row

def chunkStream2(x,cwidth = 500, stepsize = 100): 
    nx = 0
    for i in range(np.floor(cwidth/stepsize).astype(int)):
        nstart = i*stepsize
        nwide = (len(x)-nstart)
        nblock = nwide/cwidth      
        nx = int(nx + nblock)      
    xr = np.empty((nx,cwidth))  
    nstart = 0
    for i in range(nx):
        xr[i,:] = x[nstart:nstart+cwidth]
        nstart = nstart + stepsize
    return xr

def find(x):
    f = np.nonzero(x)
    if len(f)==1:
        f=f[0]
    return(f)
    
def autoencoder_plot(x,nplot,title):
    plt.figure()
    pred=autoencoder.predict(x)
    plt.subplot(2,1,1)
    plt.plot(x[nplot,:])
    plt.plot(pred[nplot,:])
    plt.legend(['Data','Reconstructed'])
    plt.title(title,fontsize=16,fontweight='bold')
    plt.subplot(2,1,2)
    plt.plot(pred[nplot,:]-x[nplot,:],'k')
    plt.title('Reconstruction error',fontsize=16,fontweight='bold')

#%% Seperate training and validation data, and scale the data based ONLY on 
# the "normal" data
d =preprocessing.StandardScaler().fit_transform(np.reshape(d1,(len(d1),1)))
d=np.reshape(d,(len(d),))
normal=chunkStream2(d)
nx = int(normal.shape[0]*.9) 
# split into training and validation sets
n_train = normal[0:nx,:]
n_val = normal[nx:,:]

#scale data using standard scalar
d21 = preprocessing.StandardScaler().fit_transform(np.reshape(d2,(len(d2),1)))
d21=np.reshape(d21,(len(d21),))
d34 = preprocessing.StandardScaler().fit_transform(np.reshape(d3,(len(d3),1)))
d34=np.reshape(d34,(len(d34),))

l_faulty = chunkStream2(d21) #less faulty
faulty = chunkStream2(d34) # sever faulty

# Here we further seperating the normal data into training data into a training and test set 
trainFrac=0.9
r = np.random.uniform(size=[len(n_train)])

X_train = n_train[find(r<trainFrac),:]
X_test = n_train[find(r>=trainFrac),:]

#%% Keras deep network model
ncol = n_train.shape[1]
first = 250
second = 77
third = 33
encoding_dim = 5

input_dim = Input(shape = (n_train.shape[1], ))

# DEFINE THE ENCODER LAYERS
encoded1 = Dense(first, activation = 'relu')(input_dim)
encoded2 = Dense(second, activation = 'relu')(encoded1)
encoded3 = Dense(third, activation = 'relu')(encoded2)

encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

# DEFINE THE DECODER LAYERS
decoded1 = Dense(third, activation = 'relu')(encoded4)
decoded2 = Dense(second, activation = 'relu')(decoded1)
decoded3 = Dense(first, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'linear')(decoded3)

# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(inputs = input_dim, outputs = decoded4)

# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'SGD', loss = 'mean_squared_error')
#autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')
AE = autoencoder.fit(X_train, X_train, epochs = 150, batch_size = 33, 
                shuffle = True, validation_data = (X_test, X_test))

#%% Plotting performance of autoencoder
# Visualize loss history
training_loss = AE.history['loss']
test_loss = AE.history['val_loss']

plt.figure()
plt.plot(training_loss, 'r--')
plt.plot(test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('encoding_dim=' + str(encoding_dim))

#%% Evaluate performance on validation set of normal dataset 
# pick a random row to plot
nplot = 2
autoencoder_plot(normal,nplot,'Normal Data')
# Evaluate performance on slightly faulted dataset
autoencoder_plot(l_faulty,nplot,'Slightly Faulty data')
# Evaluate performance on severly faulty dataset
autoencoder_plot(faulty,nplot,'Severly Faulty data')

#%% Calculate MSRE statistic and plotting distribution
n_pred_test = autoencoder.predict(X_test)
l_faulty_pred = autoencoder.predict(l_faulty)
faulty_pred = autoencoder.predict(faulty)

mse_n = np.mean(np.power(X_test - n_pred_test, 2), axis=1)
mse_l_faulty = np.mean(np.power(l_faulty - l_faulty_pred, 2), axis=1)
mse_faulty = np.mean(np.power(faulty - faulty_pred, 2), axis=1)

plt.figure()
sns.distplot(mse_n,kde= True,color = 'blue')
sns.distplot(mse_l_faulty,kde=True,color='green')
sns.distplot(mse_faulty,kde=True,color='red')
plt.xlabel('mean square reconstruction error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['normal','less faulty','severly faulty'])
plt.title('Density Function',fontsize=16,fontweight='bold')

#%% calculate Mahalanobis distance
train_error = np.reshape((X_test - mse_n),(len(X_test),1))
mean = np.mean(train_error)
cov = 0
for e in train_error:
    cov += np.dot((e-mean).reshape(len(e), 1), (e-mean).reshape(1, len(e)))
cov /= len(train_error)

def Mahala_distantce(x,mean,cov):
    d = (x - mean)**2 / cov
    return d

m_dist = []
test_error=  np.reshape((x_test[:,1] - n_pred_test[:,1]),(len(x_test),1))
for e in test_error:
    m_dist.append(Mahala_distantce(e,mean,cov))

m_dist=[float(i) for i in m_dist]
plt.plot(m_dist)
plt.figure()
sns.distplot(np.square(m_dist),bins = 5,kde= True,color = 'green');
plt.xlabel('Mahalanobis dist')
#%% Feature Extraction
# Extracted feature for normal dataset
encoder = Model(inputs = input_dim, outputs = encoded4)
features = encoder.predict(normal)
features = pd.DataFrame(features, columns = ['F1', 'F2', 'F3', 'F4','F5'])
pd.plotting.scatter_matrix(features, alpha = 0.2, diagonal = 'kde')
plt.suptitle('Features from normal data',fontweight='bold',fontsize=16)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])
hidden_representation.add(autoencoder.layers[4])

features = hidden_representation.predict(normal)

