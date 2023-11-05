import tensorflow as tf
import keras_tuner as kt
import numpy as np
import h5py as h5
import nn
from customs.losses import focal_loss
from customs.my_metrics import Expos_on_Suppr
from customs.ds_making import make_dataset


class TransformerHyperModel(kt.HyperModel):

    def __init__(self, batch_size=64, lr_opt=True, lr=0.005, decay_rate=0.9, weight_in_loss=2., default_hps=None):
        super().__init__()

        if default_hps is None:
            default_hps = {"num_heads": [8, 8],
                           "key_dims": [16, 16],
                           "value_dims": [16, 16],
                           "ff_dims": [64, 64],
                           "out_dense_units": 64,
                           "pos_encoding": True}
        self.default_hps = default_hps
        self.bs = batch_size
        self.lr_opt = lr_opt
        self.lr = lr
        self.decay_rate = decay_rate
        self.weight_in_loss = weight_in_loss

    def customise_HP(self, hp):
        if self.lr_opt:
            self.lr = hp.Float('lr_i', 1e-5, 0.1, step=10, sampling="log")
            self.decay_rate = 0.9
            self.hps_dict = self.default_hps
        else:
            self.hps_dict = {"num_heads": [hp.Choice(f'num_heads_0', [4, 32]),hp.Choice(f'num_heads_1', [4, 32])],
                             "key_dims": [hp.Choice(f'key_dims_0', [5, 32]),hp.Choice(f'key_dims_1', [5, 32])],
                             "value_dims": [hp.Choice(f'value_dims_0', [5, 32]), hp.Choice(f'value_dims_1', [5, 32])],
                             "ff_dims": [hp.Choice(f'ff_dims_0', [16, 128]),hp.Choice(f'ff_dims_1', [16, 128])],
                             "out_dense_units": hp.Choice(f'out_dense_units', [16, 128]),
                             "pos_encoding": hp.Choice(f'pos_encoding', [False, True])}

    def build(self, hp, Shape=(None, 6)):
        self.customise_HP(hp)
        model = nn.nn_transformer_classifier(Shape, **self.hps_dict)
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=10000,
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


def tune(hm_model=TransformerHyperModel, path_to_h5=None, model_name="Not_a_name", regime="lr", batch_size=64,
         shape=(None, 6),
         cutting=10,
         num_of_epochs=1,
         max_lr_trials=50, max_hp_trials=100,
         project_name_lr="tune_lr", project_name_hp="tune_hp"):
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
                     batch_size=batch_size),
            objective=kt.Objective("val_loss", direction="min"),
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
                     batch_size=batch_size),
            objective=kt.Objective("val_loss", direction="min"),
            overwrite=False,
            directory='./trained_models/' + model_name + '/tuning/',
            project_name=project_name_lr
        )
        best_from_lr_tune = tuner_lr.get_best_hyperparameters()[0].values
        best_lr = np.round(best_from_lr_tune['lr_i'], 4)
        print("LOADED BEST LR=",best_lr)

        tuner_hp = kt.RandomSearch(
            hm_model(lr_opt=False, lr=best_lr,
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
