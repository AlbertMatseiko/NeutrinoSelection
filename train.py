import tensorflow as tf
import h5py as h5
import matplotlib.pyplot as plt
from customs.my_metrics import Expos_on_Suppr
from customs.losses import focal_loss
from datetime import datetime
from customs.ds_making import make_dataset


def train_model(model, path_to_h5, batch_size, lr_initial, model_name, shape, w_mu=10., w_nu=1., num_of_epochs=200, verbose=0, cutting=1):
    with h5.File(path_to_h5, 'r') as hf:
        total_num = hf['train/ev_ids_corr/data'].shape[0]
        steps_per_epoch = (total_num // batch_size) // cutting
    print(steps_per_epoch)
    # num_of_epochs = 20z
    decay_rate = 0.05 ** (1 / num_of_epochs)
    decay_steps = steps_per_epoch

    lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_initial, decay_steps=decay_steps,
                                                        decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer=optimizer, loss=focal_loss(2., 2., w_mu, w_nu),
                  weighted_metrics=[],
                  metrics=[Expos_on_Suppr(name="expos_on_suppr", max_suppr_value=5e-6, num_of_points=100000),
                           'accuracy'])

    # Define the Keras TensorBoard callback.
    logdir = "./trained_models/logs_tb/" + model_name + "/fit/" + datetime.now().strftime(
        "%Y%m%d-%H%M%S")  # сделать общую папку logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=1e-6),
                 tf.keras.callbacks.ModelCheckpoint(filepath='./trained_models/' + model_name + '/best',
                                                    monitor='val_loss', verbose=verbose,
                                                    save_best_only=True, mode='min'), tensorboard_callback]
    train_dataset = make_dataset(path_to_h5, 'train', batch_size, shape)
    test_dataset = make_dataset(path_to_h5, 'test', batch_size, shape)

    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=num_of_epochs,
                        validation_data=test_dataset,
                        callbacks=callbacks, verbose=verbose)
    model.save('./trained_models/' + model_name + '/last')

    return history


# Рисуем процесс обучения
def make_train_figs(history, model_name):
    train_acc = history.history['loss']
    test_acc = history.history['val_loss']

    fig = plt.figure(figsize=(20, 10))
    plt.plot(train_acc, label='Training')
    plt.plot(test_acc, label='Validation')
    plt.xlabel('Sub-epoch number', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training process', fontsize=20)
    plt.legend(fontsize=16, loc=2)
    plt.grid(ls=':')
    plt.savefig('./trained_models/' + model_name + '/training.png')
    plt.close(fig)
