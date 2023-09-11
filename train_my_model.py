import h5py as h5
import tensorflow as tf
import os

# scripts for NN
import sys

sys.path.append('./nn_builds')
from nn_builds.nn import *
from training.train import train_model, make_train_figs

# GPU on
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# data and model's names
name = 'baikal_multi_0523_flat_pureMC_h5s2_norm.h5'
path_to_h5 = '../data/' + name
mn = 'nn_main_model'
model_name = mn

# making dir for model if necessary
try:
    os.makedirs('../models/' + model_name)
    print('directory for the model is created')
except:
    print('directory for the model already exists')

# getting the shape of data
Shape = (None, 6)

# set hyperparams
lr_initial = 0.005  # tuned
batch_size = 256
# making model
model = globals()[mn](Shape)
print(model.summary())

# training model and creating figs
history = train_model(model, path_to_h5, batch_size, lr_initial, model_name, shape=Shape,
                      num_of_epochs=10, verbose=1)
make_train_figs(history, model_name)