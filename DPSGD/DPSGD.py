#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# DeepPacket
# =================
#
# Data cleaning and pre-processing empployed according to the DeepPacket paper by Lotfollahi et al.

# In[3]:


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

from tensorflow.keras.utils import to_categorical
from scipy.io import loadmat



# In[4]:


from tensorflow import keras
from IPython.display import clear_output

VISUALIZE = False

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        # update history
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(np.average(logs["loss"]))
        self.val_losses.append(np.average(logs["val_loss"]))
        self.acc.append(logs["acc"])
        self.val_acc.append(logs["val_acc"])
        self.i += 1

        if VISUALIZE:

            clear_output()

            # plot lossess
            plt.figure(1)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()

            plt.figure(2)
            plt.plot(self.x, self.acc, label="accuracy")
            plt.plot(self.x, self.val_acc, label="val_accuracy")
            plt.legend()


            plt.show();



# ## Save weights routine

# In[9]:


def save_weights(model, fname=None):

    if fname is None:
        import time
        fname = "model_%d" % int(time.time())

    s = model.to_json()
    with open(fname + ".json", "w") as f:
        f.write(s)

    model.save_weights(fname + ".h5")


# ***

# # DPSGD DeepPacket

# In[ ]:


import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

from absl import logging
from tensorflow import keras

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))


# ## Flags Definition

# In[ ]:


class Flags(object):
    def __init__(self, **kwargs):
        self.dpsgd = True
        self.learning_rate = 0.15
        self.noise_multiplier = 1.1
        self.l2_norm_clip = 1
        self.batch_size = 250
        self.epochs = 60
        self.microbatches = 250
        self.model_dir = None

        for k, v in kwargs.items():
            setattr(self, k, v)


# In[ ]:


import numpy as np
from IPython.display import clear_output

VISUALIZE = False


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        # update history
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(np.average(logs["loss"]))
        self.val_losses.append(np.average(logs["val_loss"]))
        self.acc.append(logs["acc"])
        self.val_acc.append(logs["val_acc"])
        self.i += 1

        if VISUALIZE:

            clear_output()

            # plot lossess
            plt.figure(1)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()

            plt.figure(2)
            plt.plot(self.x, self.acc, label="accuracy")
            plt.plot(self.x, self.val_acc, label="val_accuracy")
            plt.legend()


            plt.show();



# In[ ]:


def create_dpsgd_model(FLAGS):

    optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate
            )

    loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

    setattr(loss, "__name__", "CategoricalCrossentropy")

    plot_losses = PlotLosses()

    #create model
    model = Sequential()

    c = NUM_CLASSES

    #add model layers
    model.add(BatchNormalization(input_shape=(NUM_FEATURES, 1)))
    model.add(Conv1D(50, kernel_size=4, strides=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(Flatten())
    model.add(Dense(c))
    model.add(Dropout(0.05))
    model.add(Dense(c, activation='softmax'))

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model, optimizer





# ## Sequence

# In[2]:


# Data Loader Class
# ----------------------------

# In[5]:

from tensorflow.keras.utils import Sequence
from path import Path


NUM_FEATURES = 1500

type_names = ['chat', 'email', 'voip', 'streaming', 'file', 'browsing', 'p2p', 'audio']
type_map = [0, 0, 1, 2, 0, 0, 3, 4, 4, 0, 0, 2, 0, 3, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
            2, 0, 4, 3, 3, 5, 5, 6, 5, 5, 3, 3, 2, 0, 6, 1, 7, 0, 4, 2, 0, 0, 3, 4, 2,
            0, 4, 3, 3, 2, 3, 3, 5]

NUM_CLASSES = len(type_names)

class DataGen(Sequence):
    """
    Generator class for feeding large data-set of giant PCAPs parsed
    and pre-processed into multiple MAT files to the model
    """

    def __init__(self, indir, idxfilter=lambda x: True, batch_size=100):
        self.indir = Path(indir)
        self.idxfilter = idxfilter
        self.batch_size = batch_size
        self.files = [f for idx, f in enumerate(self.indir.listdir())
                if f.endswith(".mat") and idxfilter(idx)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = loadmat(self.files[idx])["packets"]

        num_entries = data.shape[0] - data.shape[0] % self.batch_size

        # separate X and Y
        X_all = data[:num_entries, 0:NUM_FEATURES]
        y_all_lowlevel = data[:num_entries, [NUM_FEATURES]]

        # convert Y to high-level
        y_all = [[type_map[y[0]]] for y in y_all_lowlevel]

        # Reformat y (labels) to one-hot encoding
        y_all_cat = to_categorical(y_all, num_classes=NUM_CLASSES)

        # reshape X
        X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

        return X_all, y_all_cat

    def on_epoch_end(self):
        pass


# # Private training

# In[3]:


# ### Fix batch size for SGD implementation

# In[ ]:


batch_size = 1000


NUM_HOSTS = 10
NUM_EPOCHS = 10

FREQ = 300


host_models = []

flags = Flags(learning_rate=0.001, epochs=1)

for i in range(NUM_HOSTS):
    # create model for each host
    np_model, _ = create_dpsgd_model(flags)
    host_models.append(np_model)


# In[ ]:

import time
st = time.time()

val_gen = DataGen(
    "data/iscx_006d.mat", 
    idxfilter=lambda x: x % FREQ >= 295, 
    batch_size=batch_size
)


for epoch in range(NUM_EPOCHS):
    
    for host in range(NUM_HOSTS):
        
        # fetch model
        np_model = host_models[host]

        # create new generator sequence
        train_gen = DataGen(
            "data/iscx_006d.mat", 
            idxfilter=lambda x: x % FREQ == host * NUM_EPOCHS,
            batch_size=batch_size
        )

        # train again for 1 epoch
        np_model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=1)

        # save snapshot
        save_weights(np_model, "dp_mesh_lite/dp_lite_snapshot_host%02d_epoch%02d_2019-11-28" % (host, epoch))


et = time.time()

# In[ ]:


print("Diff-priv training finished in %f" % (et - st))


# In[ ]:


# # Compute Epsilon

# In[ ]:



def compute_epsilon(host_num, epochs, batch_size):
    """
    Computes epsilon value for given hyperparameters.
    """
    
    steps = len(gen) * epochs

    if FLAGS.noise_multiplier == 0.0:
        return float('inf')

    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = 1 / len(gen)
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=FLAGS.noise_multiplier,
        steps=steps,
        orders=orders
    )

    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]




# In[6]:


gen = DataGen(
    "data/iscx_006d.mat", 
    idxfilter=lambda x: x % FREQ == 0,
    batch_size=batch_size
)

len(gen)


# In[7]:


import os
os.getpid()


# In[ ]:




