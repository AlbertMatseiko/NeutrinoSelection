import tensorflow as tf
import tensorflow.keras.layers as tfl
from nn_constructor import *


def nn_1LSTM_3RB_1LSTM(Shape, u1, u2, f_id_list, k_id_list, f_cd_list, k_cd_list, s_cd_list):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True, так как хотим использовать маску.
    ######
    #1 LSTM
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x,mask_in, f_id = f_id_list[0], k_id = k_id_list[0], f_cd = f_cd_list[0], k_cd = k_cd_list[0], s_cd = s_cd_list[0])
    ######
    #3
    x, mask = res_block(x,mask, f_id = f_id_list[1], k_id = k_id_list[1], f_cd = f_cd_list[1], k_cd = k_cd_list[1], s_cd = s_cd_list[0])
    ######
    #4
    x, mask = res_block(x,mask, f_id = f_id_list[2], k_id = k_id_list[2], f_cd = f_cd_list[2], k_cd = k_cd_list[2], s_cd = s_cd_list[0])
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x) 
    model = tf.keras.Model(inputs=x_in, outputs=outputs) 
    return model