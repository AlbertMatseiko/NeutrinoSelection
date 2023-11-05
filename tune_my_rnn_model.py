# Basic imports
import os
import keras_tuner as kt
import nn
import rnn_tune_proc as TP


import tensorflow as tf # tensorflow and GPU
gpus = tf.config.list_physical_devices('GPU')
print("The gpu' are:")
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# data and model's names
data_names = [n for n in os.listdir('data/') if n.endswith('.h5')]
for i, h5n in enumerate(data_names):
    print(str(i + 1), ". " + h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i - 1]
path_to_h5 = './data/' + name

model_names = [n for n in dir(nn) if n.startswith('nn')]
for i, mn in enumerate(model_names):
    print(str(i + 1), ". " + mn)
i = int(input("Which model do you want to tune? Print it's number! \n"))
model_name = model_names[i - 1]
try:
    os.makedirs('./trained_models/' + model_name + '/tuning')
    print('directory for tuning is created')
except:
    print('directory for tuning already exists')

# the tuning
batch_size = 256
N = 2
u_lists = [[16, 16, 0, 0], [16, 128, 0, 0], [128, 16, 0, 0], [128, 128, 0, 0], [4, 4, 0]]
i = int(input("Print a number from 1 to 4 -- which scale of units' number to use."))
path_to_report = './trained_models/' + model_name + '/tuning/'
project_name_lr = "tune_lr_" + str(i)
project_name_hp = "tune_hp_" + str(i)
u_default = u_lists[i - 1]
num_of_epochs = 1 + 2*((i-1)//2)
_ = TP.tune(TP.LstmHyperModel, path_to_h5=path_to_h5, model_name=model_name, regime='lr',
            batch_size=batch_size, shape=(None, 6), num_of_epochs=num_of_epochs, cutting=10, max_lr_trials=20,
            project_name_lr=project_name_lr,
            project_name_hp=project_name_hp,
            u_default=u_default)
_ = TP.tune(TP.LstmHyperModel, path_to_h5=path_to_h5, model_name=model_name, regime='hp',
            batch_size=batch_size, shape=(None, 6), num_of_epochs=num_of_epochs, cutting=10, max_hp_trials=50,
            project_name_lr=project_name_lr,
            project_name_hp=project_name_hp,
            u_default=u_default)
print(_)
tuner_lr = kt.RandomSearch(
    TP.LstmHyperModel(lr_opt=True,
                      u_default=u_default,
                      N=N,
                      batch_size=batch_size),
    objective=kt.Objective("val_Expos_on_Suppr", direction="max"),
    overwrite=False,
    directory='./trained_models/' + model_name + '/tuning/',
    project_name=project_name_lr
)
best_from_lr_tune = tuner_lr.get_best_hyperparameters()[0].values
# best_lr = np.round(best_from_lr_tune['lr_i'], 4)
best_lr_metric = tuner_lr.oracle.get_best_trials(1)[0].score

tuner_hp = kt.RandomSearch(
    TP.LstmHyperModel(lr_opt=False,
                      u_default=u_default,
                      N=N,
                      batch_size=batch_size),
    objective=kt.Objective("val_Expos_on_Suppr", direction="max"),
    overwrite=False,
    directory='./trained_models/' + model_name + '/tuning/',
    project_name=project_name_hp
)
best_from_hp_tune = tuner_hp.get_best_hyperparameters()[0].values
best_hp_metric = tuner_hp.oracle.get_best_trials(1)[0].score

report_file = open(path_to_report + "/info_tune_" + str(i) + ".txt", "w")
report_file.write(f"Default units were: {u_default}. \n")
report_file.write(f"Best lr = {best_from_lr_tune} with metric = {best_lr_metric}. \n")
report_file.write(f"Best hp = {best_from_hp_tune} with metric = {best_hp_metric}. \n")
report_file.close()
