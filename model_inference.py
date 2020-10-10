import tensorflow as tf
from tensorflow import keras


def unet3d():
    in_layer = tf.keras.layers.Input((None, None, None, 1))
    bn = tf.keras.layers.BatchNormalization()(in_layer)
    cn1 = tf.keras.layers.Conv3D(8, 
                kernel_size = (1, 5, 5), 
                padding = 'same',
                activation = 'relu')(bn)
    cn2 = tf.keras.layers.Conv3D(8, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(cn1)
    bn2 = tf.keras.layers.BatchNormalization()(cn2)      
    bn2 = tf.keras.layers.Activation('relu')(bn2)

    dn1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(bn2)
    cn3 = tf.keras.layers.Conv3D(16, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(dn1)
    bn3 = tf.keras.layers.BatchNormalization()(cn3)
    bn3 = tf.keras.layers.Activation('relu')(bn3)

    dn2 = tf.keras.layers.MaxPooling3D((1, 2, 2))(bn3)
    cn4 = tf.keras.layers.Conv3D(32, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(dn2)
    bn4 = tf.keras.layers.BatchNormalization()(cn4)
    bn4 = tf.keras.layers.Activation('relu')(bn4)

    up1 = tf.keras.layers.Conv3DTranspose(16, 
                        kernel_size = (3, 3, 3),
                        strides = (1, 2, 2),
                        padding = 'same')(bn4)

    cat1 = tf.keras.layers.concatenate([up1, bn3])

    up2 = tf.keras.layers.Conv3DTranspose(8, 
                        kernel_size = (3, 3, 3),
                        strides = (2, 2, 2),
                        padding = 'same')(cat1)

    pre_out = tf.keras.layers.concatenate([up2, bn2])

    pre_out = tf.keras.layers.Conv3D(1, 
                kernel_size = (1, 1, 1), 
                padding = 'same',
                activation = 'sigmoid')(pre_out)

    pre_out = tf.keras.layers.Cropping3D((1, 2, 2))(pre_out) # avoid skewing boundaries
    out = tf.keras.layers.ZeroPadding3D((1, 2, 2))(pre_out)
    
    model = tf.keras.models.Model(inputs = [in_layer], outputs = [out])
    model.summary()
    
    return model


def run_inference(weights_path):
    model = unet3d()
    model.load_weights(weights_path)
    return


def load_data():
    return


if __name__ == "__main__":
    run_inference("convlstm_model_best_weights.hdf5")

