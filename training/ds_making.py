import h5py as h5
import numpy as np
import tensorflow as tf


# gauss noise
# 0.1 ~ 1 p.e., 150 ns, 4, 4, 15 m
apply_add_gauss = True
# normal
g_add_stds = [0.03, 0.005, 0.005, 0.005, 0.00003]
# bigger
#stds_gauss = [0.03, 0.005, 0.02, 0.02, 0.0007]
apply_mult_gauss = True
q_noise_fraction = 0.1

# limiting Q vals
set_up_Q_lim = True
up_Q_lim = 100

# mult noise
class gauss_mult_noise:
    def __init__(self, Q_mean_noise, n_fraction):
        self.Q_mean_noise = Q_mean_noise
        self.n_fraction = n_fraction
    def make_noise(self, Qs):
        noises = np.random.normal( scale=self.n_fraction, size=Qs.shape )
        Qs = Qs + (Qs+self.Q_mean_noise)*noises
        return Qs

# addative noise
def add_gauss(data, g_add_stds, ev_starts):
    g_add_stds = np.broadcast_to( g_add_stds, data.shape )
    noise = np.random.normal( scale=g_add_stds, size=data.shape )
    data += noise
    ev_idxs_local = ev_starts - ev_starts[0]
    sort_idxs = np.concatenate(
            [ np.argsort( data[ev_idxs_local[i]:ev_idxs_local[i+1],1], axis=0)+ev_idxs_local[i] for i in range(len(ev_starts)-1) ]
    )
    data = data[sort_idxs]
    return data

class generator:
    def __init__(self, file, regime, batch_size, 
                 set_up_Q_lim, up_Q_lim, 
                 apply_add_gauss, g_add_stds, apply_mult_gauss, q_noise_fraction):
        
        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        
        self.hf = h5.File(self.file,'r')
        hf = self.hf

        self.ev_starts = hf[regime + '/ev_starts/data']
        self.num = len(self.ev_starts[1:] - self.ev_starts[0:-1])
        self.batch_num = self.num // self.batch_size
        self.gen_num = self.batch_num * self.batch_size
        
        self.norm_zeros = (0.-hf['norm_param/mean'][:])/hf['norm_param/std'][:]

        #For noise
        self.set_up_Q_lim = set_up_Q_lim
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_add_stds = g_add_stds
        if set_up_Q_lim:
            Q_mean = hf['norm_param/mean'][0]
            Q_std = hf['norm_param/std'][0]
            self.Q_up_lim_norm = (up_Q_lim-Q_mean)/Q_std
        if apply_mult_gauss:
            Q_mean = hf['norm_param/mean'][0]
            Q_std = hf['norm_param/std'][0]
            self.mult_gauss = gauss_mult_noise(Q_mean/Q_std, q_noise_fraction)
        self.batch_num = self.num // self.batch_size
        
    def step(self, start, stop, ev_starts):
        hf = self.hf
        data_start = ev_starts[0]
        data_stop = ev_starts[-1]
        data = hf[self.regime + '/data/data'][data_start : data_stop]
        labels = np.zeros((self.batch_size, 2))
        ids = hf[self.regime + '/ev_ids_corr/data'][start:stop]  # id of event - starting with 'nu' or 'mu'
        ids = np.array([i[0] for i in ids]).reshape(ids.shape[0],1)
        labels[:] = np.where(ids == 110, [0,1], [1,0])  # 110 - byte code for letter 'n'
        if self.set_up_Q_lim:
            data[:,0:1] = np.where( data[:,0:1]>self.Q_up_lim_norm, self.Q_up_lim_norm, data[:,0:1] )
        # apply noise
        if self.apply_add_gauss:
            data = add_gauss(data, self.g_add_stds, ev_starts)
        if self.apply_mult_gauss:
            data[:,0] = self.mult_gauss.make_noise(data[:,0])
        check = 1
        while check:
            try:
                data = tf.RaggedTensor.from_row_starts(values=data, row_starts=ev_starts[0:-1] - ev_starts[0])
                check = 0
            except:
                check = 1
        data = data.to_tensor(default_value=self.norm_zeros)
        mask = tf.where(tf.not_equal(data[:,:,0:1], self.norm_zeros[0:1]),1.,0.)
        data = tf.concat([data, mask], axis = -1)
        
        return (data, labels)

    def __call__(self):
        start = 0
        stop = self.batch_size
        for i in range(self.batch_num):
            ev_starts = self.hf[self.regime+'/ev_starts/data'][start:stop+1]
            out_data = self.step(start, stop, ev_starts)
            np.save('./out', np.array( list(out_data[0].shape) ) )
            yield out_data
            start+=self.batch_size
            stop+=self.batch_size

def make_dataset(h5f, regime, batch_size, shape):
    bs = batch_size
    gen = generator(h5f, regime, bs, set_up_Q_lim, up_Q_lim, 
                             apply_add_gauss, g_add_stds, apply_mult_gauss, q_noise_fraction)
    dataset = tf.data.Dataset.from_generator(gen,
                                                   output_signature=(tf.TensorSpec(shape=(bs, shape[0], shape[1])),
                                                                     tf.TensorSpec(shape=(bs, 2)))
                                                  )
    if regime == 'train':
        dataset = dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset