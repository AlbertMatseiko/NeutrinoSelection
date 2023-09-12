import tensorflow as tf
import tensorflow.keras.layers as tfl
from nn_builds.nn_constructor import *

#LSTM = рекуррентные слои
#RB = Res-Net Block
#Name shows the sequence of layers
def nn_1LSTM_3RB_1LSTM_tuned1(Shape):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    u1 = 8
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = 32, k_id = 8, f_cd = 32, k_cd = 8, s_cd = 2) # f = 32, k = 12
    ######
    #3
    x, mask = res_block(x, mask, f_id = 32, k_id = 16, f_cd = 32, k_cd = 16, s_cd = 2) # f = 64, k = 10
    ######
    #4
    x, mask = res_block(x, mask, f_id = 32, k_id = 20, f_cd = 32, k_cd = 20, s_cd = 2) # f = 64, k = 8
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    u2 = 16
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x) 
    model = tf.keras.Model(inputs=x_in, outputs=outputs) 
    return model

def nn_1LSTM_3RB_1LSTM_small(Shape):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    u1 = 4
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = 32, k_id = 3, f_cd = 16, k_cd = 3, s_cd = 2) # f = 32, k = 12
    ######
    #3
    x, mask = res_block(x, mask, f_id = 32, k_id = 3, f_cd = 16, k_cd = 3, s_cd = 2) # f = 64, k = 10
    ######
    #4
    x, mask = res_block(x, mask, f_id = 32, k_id = 3, f_cd = 16, k_cd = 3, s_cd = 2) # f = 64, k = 8
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    u2 = 4
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x) 
    model = tf.keras.Model(inputs=x_in, outputs=outputs) 
    return model

def nn_main_model(Shape):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    u1 = 32
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = 64, k_id = 12, f_cd = 64, k_cd = 12, s_cd = 2) # f = 32, k = 12
    ######
    #3
    x, mask = res_block(x, mask, f_id = 128, k_id = 14, f_cd = 128, k_cd = 14, s_cd = 2) # f = 64, k = 10
    ######
    #4
    x, mask = res_block(x, mask, f_id = 64, k_id = 4, f_cd = 64, k_cd = 4, s_cd = 2) # f = 64, k = 8
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    u2 = 8
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model