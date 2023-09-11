import tensorflow as tf
import tensorflow.keras.layers as tfl

class bidir_class:
    def __init__(self, u, return_sequences = False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul'):
        self.u = u 
        self.rs = return_sequences
        self.act = act
        self.rec_act = rec_act
        self.mm = merge_mode
        
    def layer(self):
        lstm_layer = tf.keras.layers.LSTM(self.u, activation=self.act, recurrent_activation=self.rec_act, return_sequences=self.rs)
        return tf.keras.layers.Bidirectional(lstm_layer, merge_mode=self.mm)
    
    def block(self, x, mask_lstm = None):
        if mask_lstm!=None:
            #print(mask_lstm.shape,x.shape)z
            x = self.layer()(x, mask = mask_lstm)
        else:
            x = self.layer()(x)
        x = tfl.BatchNormalization(axis=-1)(x)
        return x
    
class conv_class:
    def __init__(self, filters, kernel, strides, is_change_dims = True, cd_filters = 16,
                 cd_kernel = 2, cd_strides = 2, padding = 'same', cd_padding = 'same'):
        self.f = filters 
        self.k = kernel
        self.s = strides
        self.is_cd = is_change_dims
        self.cd_f = cd_filters
        self.cd_k = cd_kernel
        self.cd_s = cd_strides
        self.pad = padding
        self.cd_pad = cd_padding
        
    def conv_id_layer(self):
        layer = tfl.Conv1D(filters=self.f, kernel_size=self.k, bias_initializer='glorot_uniform', 
                           strides=self.s, padding=self.pad)
        return layer
    
    def conv_cd_layer(self):
        layer = tfl.Conv1D(filters=self.cd_f, kernel_size=self.cd_k, bias_initializer='glorot_uniform', 
                           strides=self.cd_s, padding=self.cd_pad)
        return layer
    
    def block_id_maxpool(self, x, mask = None):
        x = self.conv_id_layer()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1])(x)
        if mask!=None:
            x = x*mask
        x = tfl.BatchNormalization(axis=-1)(x)
        if self.is_cd:
            x = tfl.MaxPooling1D(self.cd_k, strides=self.cd_s, padding = self.cd_pad)(x)
        if mask != None and self.is_cd:
            mask = tfl.MaxPooling1D(self.cd_k, strides=self.cd_s, padding = self.cd_pad)(mask)
        elif (mask != None) and (not self.is_cd):
            print('Impossible to use mask - max pooling is not defined!')
        x = tfl.BatchNormalization(axis=-1)(x)
        return x, mask

    def block_cd(self, x, mask = None):
        x = self.conv_cd_layer()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1])(x)
        x = tfl.BatchNormalization(axis=-1)(x)
        if mask != None and self.is_cd:
            mask = tfl.MaxPooling1D(self.cd_k, strides=self.cd_s, padding = self.cd_pad)(mask)
            x = x*mask
        elif (mask != None) and (not self.is_cd):
            print('Impossible to use mask - max pooling is not defined!')
        x = tfl.BatchNormalization(axis=-1)(x)
        if mask!= None:
            return x, mask
        else:
            return x
    
    def block_id_cd(self, x, mask = None):
        x = self.conv_id_layer()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1])(x)
        if mask!=None:
            x = x*mask
        x = tfl.BatchNormalization(axis=-1)(x)
        if self.is_cd:
            x = self.conv_cd_layer()(x)
            x = tf.keras.layers.PReLU(shared_axes=[1])(x)
        if mask != None and self.is_cd:
            mask = tfl.MaxPooling1D(self.cd_k, strides=self.cd_s, padding = self.cd_pad)(mask)
            x = x*mask
        elif (mask != None) and (not self.is_cd):
            print('Impossible to use mask - max pooling is not defined!')
        x = tfl.BatchNormalization(axis=-1)(x)
        if mask!= None:
            return x, mask
        else:
            return x

#Блок resnet
#id - параметры свёртки без изменения размерности
#cd - с изменением размерности
#Свёртка остаточной связи = свёртка cd.        
def res_block(x,mask, f_id,k_id, f_cd,k_cd,s_cd):
    conv = conv_class(f_id, (k_id), 1, is_change_dims = True, cd_filters = f_cd,
                 cd_kernel = k_cd, cd_strides = s_cd)
    x1, mask = conv.block_id_cd(x,mask)
    x_skip = conv.block_cd(x)
    x_skip = x_skip*mask
    x = tfl.Concatenate(axis=-1)([x1, x_skip])
    return x, mask