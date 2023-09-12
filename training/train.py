from datetime import datetime
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow as tf
from training.ds_making import make_dataset
import training.losses

def train_model(model, path_to_h5, batch_size, lr_initial, model_name, shape, num_of_epochs = 200, verbose = 0):
    with h5.File(path_to_h5, 'r') as hf:
        total_num = hf['train/ev_ids_corr/data'].shape[0]
        steps_per_epoch = (total_num // batch_size) // 1
    Shape = shape
    print(steps_per_epoch)
    #num_of_epochs = 20z
    decay_rate = 0.05 ** (1 / num_of_epochs)
    decay_steps = steps_per_epoch

    lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_initial, decay_steps=decay_steps,
                                                        decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer=optimizer, loss=training.losses.focal_loss(2., 2., 10., 1.),
                  weighted_metrics=[],
                  metrics=[tf.keras.metrics.Recall(class_id=1, name='E_0.5', dtype = tf.float64),
                           tf.keras.metrics.Recall(class_id=0, name='1-S_0.5', dtype = tf.float64),
                           tf.keras.metrics.Recall(class_id=1, thresholds = 0.9, name='E_0.9', dtype = tf.float64),
                           tf.keras.metrics.Recall(class_id=0, thresholds = 1.-0.9, name='1-S_0.9', dtype = tf.float64),
                           'accuracy'])


    # Define the Keras TensorBoard callback.
    logdir = "../trained_models/logs_tb/"+model_name+"/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") #сделать общую папку logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_1-S_0.5', mode='max', patience=5, min_delta=1e-7),
                 tf.keras.callbacks.ModelCheckpoint(filepath='../trained_models/'+model_name + '/best', monitor='val_1-S_0.5', verbose=verbose,
                                                    save_best_only=True, mode='max'), tensorboard_callback]
    train_dataset = make_dataset(path_to_h5, 'train', batch_size, Shape)
    test_dataset = make_dataset(path_to_h5, 'test', batch_size, Shape)

    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=num_of_epochs,
                        validation_data=test_dataset,
                        callbacks=callbacks, verbose=verbose)
    model.save('../trained_models/'+ model_name + '/last')

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
    plt.savefig('../figures/' + model_name + '.png')
    plt.close(fig)
