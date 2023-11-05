import tensorflow as tf
import keras_tuner as kt
import numpy as np
import h5py as h5
import nn
from customs.losses import focal_loss
from customs.my_metrics import Expos_on_Suppr
from customs.ds_making import make_dataset


class LstmHyperModel(kt.HyperModel):

    def __init__(self, batch_size=256, lr_opt=True, lr=0.005, decay_rate=0.9, weight_in_loss=10., N=2,
                 u_default=None):
        super().__init__()
        if u_default is None:
            u_default = [16, 16, 16, 16]
        self.bs = batch_size
        self.lr_opt = lr_opt
        self.lr = lr
        self.decay_rate = decay_rate
        self.weight_in_loss = weight_in_loss
        self.N = N
        self.act = 'tanh'
        self.rec_act = 'sigmoid'
        self.merge_mode = 'mul'

        self.u_default = u_default
        self.u_list = self.u_default
        self.min_units = [u // 2 for u in self.u_default]
        self.max_units = [u * 2 for u in self.u_default]
        self.step_units = [(umax - umin) // 8 for umax, umin in zip(self.max_units, self.min_units)]
        for i, s in enumerate(self.step_units):
            if s == 0:
                self.step_units[i] = 1

        self.is_mask = True

    def customise_HP(self, hp):
        if self.lr_opt:
            self.lr = hp.Float('lr_i', 1e-4, 1, step=10, sampling="log")
            self.decay_rate = 0.9  # hp.Choice('decay_rate', [0.8, 0.9, 0.95])
            self.u_list = self.u_default
            self.weight_in_loss = 10.  # hp.Choice('loss_w', [8., 10., 12., 15.])
        else:
            self.u_list = []
            for i in range(self.N):
                self.u_list.append(hp.Int('u' + str(i + 1),
                                          min_value=self.min_units[i], max_value=self.max_units[i],
                                          step=self.step_units[i]))
            # self.act = hp.Choice('act', ['tanh', 'sigmoid'])
            # self.rec_act = hp.Choice('rec_act', ['tanh', 'sigmoid'])
            # self.merge_mode = hp.Choice('merge_mode', ['concat', 'mul'])
            # self.is_mask = hp.Choice('mask', [True, False])

    def build(self, hp, Shape=(None, 6)):
        self.customise_HP(hp)
        model = nn.nn_rnn_model(Shape, self.N, self.u_list, self.act, self.rec_act, self.merge_mode, self.is_mask)
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=1000,
            decay_rate=self.decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             amsgrad=False, name='Adam')

        model.compile(optimizer=optimizer, loss=focal_loss(2., 2., self.weight_in_loss, 1.),
                      metrics=[Expos_on_Suppr(name='Expos_on_Suppr', max_suppr_value=1e-6, num_of_points=100000),
                               'accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=self.bs,
                         verbose=True, **kwargs)


def tune(hm_model=LstmHyperModel, path_to_h5=None, model_name="Not_a_name", regime="lr", batch_size=256,
         shape=(None, 6), N=2,
         cutting=10,
         num_of_epochs=1,
         max_lr_trials=50, max_hp_trials=100,
         project_name_lr="tune_lr", project_name_hp="tune_hp",
         u_default=None):
    if u_default is None:
        u_default = [16, 16, 16, 16]
    assert path_to_h5 is not None
    with h5.File(path_to_h5, 'r') as hf:
        total_num = hf['train/ev_ids_corr/data'].shape[0]
        steps_per_epoch = (total_num // batch_size) // cutting

    train_data = make_dataset(path_to_h5, regime="train", batch_size=batch_size, shape=shape)
    test_data = make_dataset(path_to_h5, regime="train", batch_size=batch_size, shape=shape,
                             start=int(steps_per_epoch * batch_size))

    # tune my learning rate with loss parameters
    if regime == "lr":
        tuner = kt.RandomSearch(
            hm_model(lr_opt=True,
                     u_default=u_default,
                     batch_size=batch_size,
                     N=N),
            objective=kt.Objective("val_Expos_on_Suppr", direction="max"),
            max_trials=max_lr_trials,
            overwrite=True,
            directory='./trained_models/' + model_name + '/tuning/',
            project_name=project_name_lr,
            max_retries_per_trial=0,
        )
        _ = tuner.search(train_data, epochs=num_of_epochs, steps_per_epoch=steps_per_epoch,
                         validation_data=test_data, validation_steps=int(3 * 1e6 / batch_size))
        return tuner.get_best_hyperparameters()
    else:
        tuner_lr = kt.RandomSearch(
            hm_model(lr_opt=True,
                     u_default=u_default,
                     N=N,
                     batch_size=batch_size),
            objective=kt.Objective("val_Expos_on_Suppr", direction="max"),
            overwrite=False,
            directory='./trained_models/' + model_name + '/tuning/',
            project_name=project_name_lr
        )
        best_from_lr_tune = tuner_lr.get_best_hyperparameters()[0].values
        best_lr = np.round(best_from_lr_tune['lr_i'], 4)

        tuner_hp = kt.RandomSearch(
            hm_model(lr_opt=False, lr=best_lr,
                     u_default=u_default,
                     N=N,
                     batch_size=batch_size),
            objective=kt.Objective("val_Expos_on_Suppr", direction="max"),
            max_trials=max_hp_trials,
            overwrite=True,
            directory='./trained_models/' + model_name + '/tuning/',
            project_name=project_name_hp
        )
        _ = tuner_hp.search(train_data, epochs=num_of_epochs, steps_per_epoch=steps_per_epoch,
                            validation_data=test_data, validation_steps=int(2.5 * 1e6 / batch_size))
    return _
