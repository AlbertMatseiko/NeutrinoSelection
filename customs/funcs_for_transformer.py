import tensorflow as tf

Concatenate = tf.keras.layers.Concatenate


def positional_encoding(length, depth):
    depth_in = depth / 2
    positions = tf.range(length)[:, tf.newaxis]  # (seq, 1)
    depths = tf.range(depth_in)[tf.newaxis, :] / depth_in  # (1, depth)
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = tf.cast(positions, tf.float32) * tf.cast(angle_rates, tf.float32)  # (pos, depth)

    pos_encoding = Concatenate(axis=-1)(
        [tf.sin(angle_rads), tf.cos(angle_rads)]
    )

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_att_mask(mask):
    arr = tf.ones(tf.shape(mask), dtype=tf.bool)
    mask_expand = arr & tf.transpose(mask, perm=[0, 2, 1])
    att_mask = mask_expand & tf.transpose(mask_expand, perm=[0, 2, 1])
    return att_mask
