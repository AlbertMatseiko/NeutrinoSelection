from customs import nn_constructor as nc
from customs.funcs_for_transformer import *
import tensorflow as tf
from tensorflow import keras as k

tfl = k.layers
Model = k.models.Model
Conv1D = tfl.Conv1D
MultiHeadAttention = tfl.MultiHeadAttention
Dense = tfl.Dense
Dropout = tfl.Dropout
Concatenate = tfl.Concatenate
LayerNormalization = tfl.LayerNormalization
GlobalAveragePooling1D = tfl.GlobalAveragePooling1D
Input = tfl.Input
Add = tfl.Add
LeakyReLU = tfl.LeakyReLU
AveragePooling1D = tfl.AveragePooling1D

SEED = 42
GU = tf.keras.initializers.GlorotUniform(seed=SEED)


def nn_transformer_classifier(input_shape, num_heads=None, key_dims=None,
                              value_dims=None, num_transformer_blocks=2,
                              ff_depth=2, ff_dims=None,  # DO NOT CHANGE STRIDES
                              out_dense_units=64, mlp_dropout=0., pos_encoding=True, YOUR_NUM_CLASSES=2,
                              MAX_LENGTH=1000):
    if ff_dims is None:
        ff_dims = [128] * 5
    if value_dims is None:
        value_dims = [32] * 5
    if key_dims is None:
        key_dims = [32] * 5
    if num_heads is None:
        num_heads = [8 * 2 ** i for i in range(5)]

    depth = input_shape[1] - 1
    inputs = Input(shape=input_shape, name='Input')

    # cut events len on MAX_LENGTH
    MAX_LENGTH = tf.cast(MAX_LENGTH, tf.int32)
    x_in = inputs[:, :MAX_LENGTH, :]  # this works, I checked

    length = tf.shape(x_in)[1]
    x = x_in[:, :, :-1]
    mask_in = x_in[:, :, -1:]
    mask = tf.cast(mask_in[:, :, 0:1], bool, name='GetMask')

    # Add positional encodings to the input data
    if pos_encoding:
        p_en = positional_encoding(length=1000, depth=depth)
        x *= tf.math.sqrt(tf.cast(depth, tf.float32, name='cast_depth'), name='norm_on_depth')
        x = Add(name='AddPosEncoding')([x, p_en[tf.newaxis, :length, :depth]])
    x._keras_mask = mask

    for i_tb in range(num_transformer_blocks):
        # expanding mask to be attention mask
        att_mask = get_att_mask(mask)
        # Multi-head self-attention with residual connection and layer normalization
        # attention mask prevents MHA from learning and counting on auxillary hits
        att = MultiHeadAttention(num_heads=num_heads[i_tb], key_dim=key_dims[i_tb], value_dim=value_dims[i_tb],
                                 name=f'MHA_{i_tb}', kernel_initializer=GU)(query=x, value=x, attention_mask=att_mask)
        x = Add(name=f'ResAdd_{i_tb}')([x, att])
        x = LayerNormalization(name=f'LayerNorm_MH{i_tb}_beforeFF')(x)

        # Feed-forward layer with residual connection and layer normalization
        assert ff_depth > 0
        for i_ff in range(ff_depth):
            ff = Conv1D(filters=ff_dims[i_ff], kernel_size=1, strides=1,
                        name=f"1DConv_MH{i_tb}_FF{i_ff}", kernel_initializer=GU)(x)
            ff = LeakyReLU(alpha=0.1, name=f'LReLU_MH{i_tb}_FF{i_ff}')(ff)
            ff = LayerNormalization(name=f'LayerNorm_MH{i_tb}_FF{i_ff}')(ff)
        # skip connections as concat
        x = Concatenate(axis=-1, name=f'ResConcat_MH{i_tb}')([x, ff])
        x = LayerNormalization(name=f'LayerNorm_MH{i_tb}_afterFF')(x)
        # добавляем нулей, чтобы правильно сделать AvPool
        x = tf.pad(x, [[0, 0], [0, 4], [0, 0]])
        mask = tf.pad(mask, [[0, 0], [0, 4], [0, 0]])
        # Считаем среднее по времени с окном 4 и шагом 2, исключая вспомогательные хиты умножением на маску.
        x = AveragePooling1D(pool_size=4, strides=2, padding='valid', data_format='channels_last',
                             name=f"AvPool_X_MH{i_tb}")(x * tf.cast(mask, tf.float32))
        mask_test = AveragePooling1D(pool_size=4, strides=2, padding='valid', data_format='channels_last',
                                     name=f"AvPool_mask_MH{i_tb}")(tf.cast(mask, tf.float32))
        mask = tf.where(tf.equal(mask_test, 0.), False, True)
        x = LayerNormalization(name=f'LayerNorm_MH{i_tb}_afterAvPool')(x)

    # Global average pooling after all transformer blocks
    x = GlobalAveragePooling1D(data_format='channels_last', name=f'GlobAv')(x, mask=mask[:, :, 0])
    x = LayerNormalization(name=f'LayerNorm_afterGAP')(x)

    # Fully connected layer with dropout
    x = Dense(out_dense_units, name=f'Dense_out_0', kernel_initializer=GU)(x)
    x = LeakyReLU(alpha=0.1, name=f'LReLU_out_0')(x)
    x = Dropout(mlp_dropout, name=f'Dropout_out_0')(x)
    x = LayerNormalization(name=f'LayerNorm_after_Dense_out_0')(x)

    num_classes = YOUR_NUM_CLASSES
    # Output layer with softmax activation for one-hot vector
    outputs = Dense(num_classes, activation='softmax', name=f'Dense_out_1', kernel_initializer=GU)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def nn_rnn_model(shape, num_of_lstm=2, u_list=None, act='tanh', rec_act='sigmoid', merge_mode='mul', mask=True):
    if u_list is None:
        u_list = [32, 16]
    assert len(u_list) >= num_of_lstm
    # input
    x_in = tf.keras.Input(shape=shape)
    if mask:
        mask_in = x_in[:, :, -1:]
        x = x_in[:, :, :-1]
        mask_lstm = tf.cast(mask_in[:, :, 0], bool)  # используем только то число шагов по времени, что есть в событии
    else:
        x = x_in[:, :, :]
        mask_lstm = None
    ######
    # Добавляем LSTM слой с return_sequences = True.
    ######

    # first LSTMs
    for i in range(int(num_of_lstm) - 1):
        u = u_list[i]
        x = nc.BidirClass(u, True, act=act, rec_act=rec_act, merge_mode=merge_mode).block(x, mask_lstm)
        # не умножаем на маску, далее она прописана
    ######
    # Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    # last LSTM
    u = u_list[int(num_of_lstm) - 1]
    bidir2 = nc.BidirClass(u, False, act=act, rec_act=rec_act, merge_mode=merge_mode)
    x = bidir2.block(x, mask_lstm)
    # + Dense
    # softmax
    outputs = tfl.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model


def nn_main_model_NoMaskAsChannel(shape, u1=32, u2=8, f_id_list=None, k_id_list=None,
                                  f_cd_list=None, k_cd_list=None, s_cd_list=None):
    # input
    if f_id_list is None:
        f_id_list = [64, 128, 64]
    if k_id_list is None:
        k_id_list = [12, 14, 4]
    if f_cd_list is None:
        f_cd_list = [64, 128, 64]
    if k_cd_list is None:
        k_cd_list = [12, 14, 4]
    if s_cd_list is None:
        s_cd_list = [2, 2, 2]
    x_input = tf.keras.Input(shape)
    mask_in = x_input[:, :, -1:]
    x_in = x_input[:, :, :-1]
    print(x_in.get_shape())
    ######
    # Добавляем LSTM слой с return_sequences = True.
    ######
    # 1 LSTM
    bidir1 = nc.BidirClass(u1, True, act='tanh', rec_act='sigmoid', merge_mode='mul')
    mask_lstm = tf.cast(mask_in[:, :, 0], bool)
    x = bidir1.block(x_in, mask_lstm) * mask_in
    ######
    # 2
    # Блок resnet
    x, mask = nc.res_block(x, mask_in, f_id=f_id_list[0], k_id=k_id_list[0],
                           f_cd=f_cd_list[0], k_cd=k_cd_list[0], s_cd=s_cd_list[0])
    ######
    # 3
    x, mask = nc.res_block(x, mask, f_id=f_id_list[1], k_id=k_id_list[1],
                           f_cd=f_cd_list[1], k_cd=k_cd_list[1], s_cd=s_cd_list[1])
    ######
    # 4
    x, mask = nc.res_block(x, mask, f_id=f_id_list[2], k_id=k_id_list[2],
                           f_cd=f_cd_list[2], k_cd=k_cd_list[2], s_cd=s_cd_list[2])
    ######
    # Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    # 5 LSTM
    bidir2 = nc.BidirClass(u2, False, act='tanh', rec_act='sigmoid', merge_mode='mul')
    mask_lstm = tf.cast(mask[:, :, 0], bool)
    x = bidir2.block(x, mask_lstm)
    # + Dense
    # softmax
    outputs = tfl.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    return model
