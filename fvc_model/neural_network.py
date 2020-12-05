from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from segmentation import models
import tensorflow as tf


def transfer_weights(source_model, target_model):
    idx = 0
    while True:
        try:
            target_model.get_layer(index=idx).set_weights(source_model.get_layer(index=idx).get_weights())
        except ValueError:
            break
        idx += 1
    return target_model


def nn_model(scan_size, backbone_weights=None, freeze_backbone=False):
    backbone = models.BCDU_net_D3(input_size=scan_size)

    if backbone_weights:
        source_model = load_model(backbone_weights)
        backbone = transfer_weights(source_model, backbone)
    if freeze_backbone:
        backbone.trainable = False

    squeezed = Lambda(lambda x: K.squeeze(x, axis=-1))(backbone.output)
    bc_1 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(squeezed)
    mp_1 = layers.MaxPooling2D(pool_size=(3, 3))(bc_1)

    bc_2 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(mp_1)
    mp_2 = layers.MaxPooling2D(pool_size=(3, 3))(bc_2)

    bc_3 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(mp_2)
    mp_3 = layers.MaxPooling2D(pool_size=(3, 3))(bc_3)

    flat = layers.Flatten()(mp_3)
    fc_1 = layers.Dense(512, activation="relu")(flat)

    categorical_input = Input(shape=(4,))
    cat_1 = Dense(32, activation="relu")(categorical_input)
    cat_2 = Dense(16, activation="relu")(cat_1)
    cat_3 = Dense(8, activation="relu")(cat_2)

    combined = layers.concatenate([fc_1, cat_3])

    output_1 = Dense(32, activation="relu")(combined)
    output_2 = Dense(1)(output_1)

    model = Model(inputs=[backbone.input, categorical_input], outputs=output_2)
    return model

def segmentation_model(scan_size, backbone_weights=None, freeze_backbone=False):
    backbone = models.BCDU_net_D3(input_size=scan_size)

    if backbone_weights:
        source_model = load_model(backbone_weights)
        backbone = transfer_weights(source_model, backbone)
    if freeze_backbone:
        backbone.trainable = False

    model = Model(inputs=[backbone.input], outputs=backbone.output)

    return model


def test_nn_model():
    backbone = models.BCDU_net_D3(input_size=(128, 64, 64, 1))
    model = Model(inputs=backbone.input, outputs=backbone.output)
    return model

