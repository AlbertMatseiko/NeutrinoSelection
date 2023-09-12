import h5py as h5
import tensorflow as tf
import os

# GPU on
gpus = tf.config.list_physical_devices('GPU')
print("The gpu' are:")
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    
# data and model's names
name = 'baikal_multi_0523_flat_pureMC_h5s2_norm_small.h5'
path_to_h5 = './data/' + name

# scripts for NN
from training.train import train_model, make_train_figs

import nn_builds.nn as nn
model_names = [n for n in dir(nn) if n.startswith('nn')]
for i,mn in enumerate(model_names):
    print(str(i+1),". "+mn)
i = int(input("Which model do you want to train? Print it's number! \n"))
model_name = model_names[i-1]

# making dir for model if necessary
try:
    os.makedirs('./trained_models/' + model_name)
    print('directory for the model is created')
except:
    print('directory for the model already exists')

# getting the shape of data
Shape = (None, 6)

# set hyperparams
lr_initial = 0.003  # tuned
batch_size = 256

# making model
from nn_builds.nn import *
model = globals()[mn](Shape)

trigger = input("Do you want to see model's summary? Type only 'y' or 'n': \n")
if trigger == 'y':
    print(model.summary())
elif trigger == 'n':
    pass
else:
    print("Your input is incorrect. Summary will not be shown.")

# settings for training
epochs = int(input("Print max number of epochs that you want in the trial: \n"))
trigger = input("Do you want to see verbose while training? Type only 'y' or 'n': \n")
if trigger == 'y':
    v = 1
elif trigger == 'n':
    v = 0
else:
    print("Your input is incorrect. Verbose will not be shown.")
    v = 0
    
# training model and creating figs
history = train_model(model, path_to_h5, batch_size, lr_initial, model_name, shape=Shape,
                      num_of_epochs=epochs, verbose=v)
make_train_figs(history, model_name)