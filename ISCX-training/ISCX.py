#!/usr/bin/env python
# coding: utf-8

# DeepPacket
# =================
# 
# Data cleaning and pre-processing empployed according to the DeepPacket paper by Lotfollahi et al.

# In[1]:


import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization

from livelossplot import PlotLossesKeras

from ann_visualizer.visualize import ann_viz


from tensorflow.keras.utils import to_categorical
from filepath.filepath import fp
from scipy.io import loadmat


# Data Loader Class
# ----------------------------

# In[2]:


from tensorflow.keras.utils import Sequence
from filepath.filepath import fp

NUM_CLASSES = 58
NUM_FEATURES = 1500

class DataGen(Sequence):
    """
    Generator class for feeding large data-set of giant PCAPs parsed
    and pre-processed into multiple MAT files to the model
    """
    
    def __init__(self, indir, idxfilter=lambda x: True, batch_per_file=100):
        self.indir = fp(indir)
        self.idxfilter = idxfilter
        self.files = [f.path() for f in self.indir.ls() if f.ext() == "mat"]
        self.batch_per_file = batch_per_file
        self.cache = None
    
    def __len__(self):
        return len([f for idx, f in enumerate(self.files) if self.idxfilter(idx)]) * self.batch_per_file
    
    def __getitem__(self, idx):
        
        file_idx = idx // self.batch_per_file
        batch_idx = idx % self.batch_per_file
        
        f = self.files[file_idx]
        
        if self.cache is not None and f == self.cache[0]:
            loaded = self.cache[1]
        else:
            loaded = loadmat(f)["packets"]
            self.cache = (f, loaded)
        
        batch_size = loaded.shape[0] // self.batch_per_file
        
        if batch_idx != self.batch_per_file - 1:
            data = loaded[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
        else:
            data = loaded[batch_idx * batch_size :, :]
        
        # separate X and Y
        X_all = data[:, 0:NUM_FEATURES]
        y_all = data[:, [NUM_FEATURES]]

        # Reformat y (labels) to one-hot encoding
        y_all_cat = to_categorical(y_all, num_classes=NUM_CLASSES)
        
        # reshape X
        X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)
        
        return X_all, y_all_cat


# In[3]:


#create model3
model3 = Sequential()

c = NUM_CLASSES

#add model3 layers
model3.add(BatchNormalization(input_shape=(NUM_FEATURES, 1)))
model3.add(Conv1D(200, kernel_size=4, strides=3, activation='relu'))
model3.add(Conv1D(200, kernel_size=5, strides=1, activation='relu'))
model3.add(MaxPooling1D(pool_size=3, strides=2))
model3.add(Flatten())
model3.add(Dense(c))
model3.add(Dropout(0.05))
model3.add(Dense(c))
model3.add(Dropout(0.05))
model3.add(Dense(c, activation='softmax'))

# compile model3
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[5]:


train_gen = DataGen("data/iscx_006d.mat", idxfilter=lambda x: x % 3 != 2, batch_per_file=1)
val_gen = DataGen("data/iscx_006d.mat", idxfilter=lambda x: x % 3 == 2, batch_per_file=1)


# In[ ]:


model3.fit_generator(generator=train_gen, validation_data=val_gen, epochs=4)


# In[70]:


from keras import backend as K

K.set_value(model3.optimizer.lr, 0.0001)

model3.fit_generator(generator=train_gen, validation_data=val_gen, epochs=10, callbacks=[PlotLossesKeras()])


# Saving the model
# -------------------------

# In[71]:


s = model3.to_json()
with open("dp_model_3.json", "w") as f:
    f.write(s)

model3.save_weights('dp_model_3.h5')


# In[4]:


model3.load_weights('dp_model_3.h5')


# Confusion Matrix
# ------------------------

# In[9]:


predictions = model3.predict_generator(val_gen)


# In[10]:


from sklearn.metrics import confusion_matrix

pred_class = predictions.argmax(axis=1)
true_class = np.concatenate([batch[1].argmax(axis=1) for batch in val_gen])

# Calculate confusion matrix
cm = confusion_matrix(true_class, pred_class)

zero = np.array([10**-32] * cm.shape[0])

# Calculate accuracy, per-class recall & precision
accuracy = cm.diagonal().sum() / cm.sum()
recall = np.divide(cm.diagonal(), cm.sum(axis=1))
precision = np.divide(cm.diagonal(), cm.sum(axis=0) + zero)

