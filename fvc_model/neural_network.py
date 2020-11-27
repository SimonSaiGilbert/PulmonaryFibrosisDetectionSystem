from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from segmentation import models



def nn_model(freeze_backbone=False):
    backbone = models.BCDU_net_D3(input_size=(None, 602, 512, 512))

    if freeze_backbone:
        backbone.trainable = False

    bc_1 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(backbone.output)
    mp_1 = layers.MaxPooling2D(pool_size=(3, 3))(bc_1)

    bc_2 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(mp_1)
    mp_2 = layers.MaxPooling2D(pool_size=(3, 3))(bc_2)

    bc_3 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(mp_2)
    mp_3 = layers.MaxPooling2D(pool_size=(3, 3))(bc_3)

    bc_4 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(mp_3)
    mp_4 = layers.MaxPooling2D(pool_size=(3, 3))(bc_4)

    flat = layers.Flatten()(mp_4)
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


def test_nn_model():
    backbone = models.BCDU_net_D3(input_size=(64, 128, 128, 1))
    model = Model(inputs=backbone.input, outputs=backbone.output)
    return model

