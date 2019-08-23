# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:14:40 2019

@author: Admin
"""
#%% Assignment anomaly detection
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
plt.close('all')
#Reading files
os.chdir('D:/NASA bearing dataset')
df=pd.read_csv('merged_dataset_BearingTest_1.csv')
cn=list(df.columns)

#%% Keras 
## input layer
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.io as spio

sns.set(style="whitegrid")
np.random.seed(203)
os.chdir('D:\Assignment')
mat1 = spio.loadmat('IGBOutput7.mat') ## Normal
mat2 = spio.loadmat('IGBOutput21.mat') ## Slight faulty
mat3 = spio.loadmat('IGBOutput34.mat') ## Severely faulty

d1 = mat1['Output'][0]
d2 = mat2['Output'][0]
d3 = mat3['Output'][0]

normal=np.reshape(d1,(len(d1),1))
l_faulty=np.reshape(d2,(len(d2),1))
faulty=np.reshape(d3,(len(d3),1))

normal = preprocessing.MinMaxScaler().fit_transform(normal)
faulty = preprocessing.MinMaxScaler().fit_transform(faulty)
l_faulty = preprocessing.MinMaxScaler().fit_transform(l_faulty)

#%% Deep Autoencoder with 3 dense layer
input_layer = Input(shape=(normal.shape[1],))
## encoding architecture
encode_layer1 = Dense(250, activation='tanh')(input_layer)
encode_layer2 = Dense(77, activation='tanh')(encode_layer1)
encode_layer3 = Dense(2, activation = 'tanh')(encode_layer2)

## decoding architecture
decode_layer1 = Dense(77, activation='tanh')(encode_layer3)
decode_layer2 = Dense(250, activation='tanh')(decode_layer1)

## output layer
output_layer  = Dense((normal.shape[1]))(decode_layer2)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="SGD", loss="mean_squared_error")
AE=autoencoder.fit(normal[:500], normal[:500],batch_size = 20, epochs = 100, 
                shuffle = False, validation_split = 0.30)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])

norm_hid_rep = hidden_representation.predict(normal[:500])
faulty_hid_rep = hidden_representation.predict(faulty[:500])
faulty_hid_rep_1 = hidden_representation.predict(l_faulty[:500])

#%% Calculating MSE or reconstruction error
normal=normal-normal[0]
norm_hid_rep=norm_hid_rep-norm_hid_rep[0]
mse = np.mean(np.abs(normal[:500]-norm_hid_rep[:500,0]), axis = 1)
plt.figure()
plt.subplot(2,1,1)
plt.plot(norm_hid_rep[:500,0])
plt.plot(normal[:500])
plt.legend(['autoencoded','good data'])
plt.subplot(2,1,2)
plt.plot(mse)
plt.legend(['reconstruction error'])

plt.figure()
faulty_hid_rep=faulty_hid_rep-faulty_hid_rep[0]
faulty=faulty-faulty[0]
mse_f= np.mean(np.abs(faulty[:500]-faulty_hid_rep[:500,0]), axis = 1)
plt.subplot(2,1,1)
plt.plot(faulty_hid_rep[:500,0])
plt.plot(faulty[:500])
plt.legend(['autoencoded','bad data'])
plt.subplot(2,1,2)
plt.plot(mse_f)
plt.legend(['reconstruction error'])

#PDF function plot
training_loss = AE.history['loss']
test_loss = AE.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure()
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('encoding_dim=' + str(1))

plt.figure()
mse_f1= np.mean(np.abs(l_faulty[:500]-faulty_hid_rep_1[:500]), axis = 1)
sns.distplot(mse,kde= True,color = 'blue')
sns.distplot(mse_f1,kde=True,color='yellow')
sns.distplot(mse_f,kde=True,color='red')
plt.legend(['Bearing-1(Faulty)','Bearing-1(Normal)','Bearing-2','Bearing-3','Bearing-4'])
plt.title('PDF with Histogram')

from scipy.stats import kde

kde1 = kde.gaussian_kde(mse)
kde21 =kde.gaussian_kde(mse_f1)
kde34 =kde.gaussian_kde(mse_f)

xx = np.linspace(0, 0.5, 300)

plt.plot(xx,kde1(xx))
plt.plot(xx,kde21(xx),color = 'orange')
plt.plot(xx,kde34(xx),color = 'green')
# =============================================================================
# plt.legend(['normal','slight faulty','severe faulty'])
# 
# =============================================================================

