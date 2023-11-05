import os
import nn
import train as tr
import tensorflow as tf

# GPU on
gpus = tf.config.list_physical_devices('GPU')
print("The gpu' are:")
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# data and model's names
data_names = os.listdir('./data/')
data_names = [n for n in data_names if n.endswith('.h5')]

for i, h5n in enumerate(data_names):
    print(str(i + 1), ". " + h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i - 1]
# name = 'baikal_multi_0523_flat_pureMC_h5s2_norm.h5'
path_to_h5 = './data/' + name

# scripts for NN

model_names = [n for n in dir(nn) if n.startswith('nn')]
for i, mn in enumerate(model_names):
    print(str(i + 1), ". " + mn)
i = int(input("Which model do you want to train? Print it's number! \n"))
model_name = model_names[i - 1]
# model_name = 'nn_rnn_model'
# making dir for model if necessary
try:
    os.makedirs('./trained_models/' + model_name)
    print('directory for the model is created')
except:
    print('directory for the model already exists')

# getting the shape of data
Shape = (None, 6)

# set hyperparams
lr_initial = 0.001  # tuned
batch_size = 256

# making model
LOAD = False
u_list = [160, 11]
w_mu = 1.
model_func = getattr(nn, model_name)
#model_name = input("How do you want to call the model/trial?")
model_name = f"{model_name}_bestHP3_wmu{int(w_mu)}"
if LOAD:
    try:
        path_to_model = './trained_models/' + model_name + '/' + 'best'
        model = tf.keras.models.load_model(path_to_model, compile=False)
    except OSError:
        print("Can't load the model, creating new")
        model = model_func(Shape, u_list=u_list)
else:
    model = model_func(Shape, u_list=u_list)  # , u_list=[32, 16])
# model_name = 'test'

trigger = input("Do you want to see model's summary? Type only 'y' or 'n': \n")
# trigger = 'n'
if trigger == 'y':
    print(model.summary())
elif trigger == 'n':
    pass
else:
    print("Your input is incorrect. Summary will not be shown.")

# settings for training
epochs = int(input("Print max number of epochs that you want in the trial: \n"))
# epochs = 2
# trigger = input("Do you want to see verbose while training? Type only 'y' or 'n': \n")
trigger = 'y'
if trigger == 'y':
    v = 1
elif trigger == 'n':
    v = 0
else:
    print("Your input is incorrect. Verbose will not be shown.")
    v = 0

# training model and creating figs
history = tr.train_model(model, path_to_h5, batch_size, lr_initial, model_name, w_mu=w_mu, shape=Shape,
                         num_of_epochs=epochs, verbose=v)
#tr.make_train_figs(history, model_name)
