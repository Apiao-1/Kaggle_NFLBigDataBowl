import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split, KFold
from keras.utils import plot_model
import numpy as np
from NFLBigDataBowl.test import test
import tensorflow as tf

keras.backend.clear_session()

def crps(y_true, y_pred):
    loss = K.mean((K.cumsum(y_pred, axis=1) - y_true) ** 2)
    return loss


def train_and_get_model_NN(train_dense_game, train_dense_players, train_cat_game, train_cat_players, train_y_raw, train_y,
              num_classes, tr_inds, val_inds, batch_size=32, epochs=10):
    ## model dense
    input_dense_game = keras.layers.Input(shape=(train_dense_game.shape[1],), name="numerical_game_feature_input")
    x1 = keras.layers.Dense(32, activation="relu", name="numerical_game_feature1")(input_dense_game)
    x1 = keras.layers.Dropout(0.5)(x1)
    # x1 = keras.layers.Dropout(0.1)(x1)

    input_dense_players = keras.layers.Input(shape=(train_dense_players.shape[1], train_dense_players.shape[2]),
                                             name="numerical_players_feature_input")
    x2 = keras.layers.Dense(32, activation="relu", name="numerical_players_feature1")(input_dense_players)
    x2 = keras.layers.Dropout(0.5)(x2)
    # x2 = keras.layers.Flatten()(x2)
    # x2 = keras.layers.Dropout(0.1)(x2)

    ## model categorical
    input_cat_game = keras.layers.Input(shape=(train_cat_game.shape[1],), name="categorical_game_feature_input")
    embedding = keras.layers.Embedding(num_classes, 8, embeddings_regularizer=regularizers.l2(1))

    x3 = embedding(input_cat_game)
    x3 = keras.layers.Flatten()(x3)
    x3 = keras.layers.Dense(8, activation="relu", name="categorical_game_feature1")(x3)
    x3 = keras.layers.Dropout(0.6)(x3)

    input_cat_players = keras.layers.Input(shape=(train_cat_players.shape[1], train_cat_players.shape[2]),
                                           name="categorical_player_feature_input")
    x4 = embedding(input_cat_players)

    x4 = keras.layers.Reshape((int(x4.shape[1]), int(x4.shape[2]) * int(x4.shape[3])))(x4)
    x4 = keras.layers.Dense(16, activation="relu", name="categorical_players_feature1")(x4)
    x4 = keras.layers.Dropout(0.6)(x4)

    ### concat players
    x_concat_players = keras.layers.Concatenate(name="players_features_concat")([x2, x4])
    x_concat_players = keras.layers.Dense(16, activation="relu", name="players_features1")(x_concat_players)
    # x_concat_players = keras.layers.GlobalAveragePooling1D()(x_concat_players)

    ## flatten
    x2 = keras.layers.Flatten()(x2)
    x4 = keras.layers.Flatten()(x4)
    x_concat_players = keras.layers.Flatten()(x_concat_players)

    ### concat all
    x_concat = keras.layers.Concatenate(name="features_concat")([x1, x3, x_concat_players] + [x2, x4])
    x_concats = []
    n_unit = 128
    decay_rate = 0.5
    for k in range(5):
        x_concat = keras.layers.Dense(n_unit, activation="relu")(x_concat)
        x_concats.append(x_concat)
        n_unit = int(n_unit * decay_rate)
    x_concat = keras.layers.Concatenate(name="deep_features")(x_concats)
    x_concat = keras.layers.Dropout(0.5)(x_concat)

    ## concat
    x_concat = keras.layers.Concatenate(name="all_concat")([x1, x3, x_concat_players, x_concat] + [x2, x4])
    out_soft = keras.layers.Dense(199, activation="softmax", name="out_soft")(x_concat)
    out_reg = keras.layers.Dense(1, activation=None, name="out_reg")(x_concat)
    model = keras.models.Model(inputs=[input_dense_game, input_dense_players, input_cat_game, input_cat_players],
                               outputs=[out_soft, out_reg])

    ## compile
    model.compile(loss=[crps, keras.losses.mae],
                  loss_weights=[1.0, 0.01],
                  optimizer=keras.optimizers.Adam(lr=0.005, decay=1e-4))

    ## train
    tr_x = [train_dense_game[tr_inds], train_dense_players[tr_inds], train_cat_game[tr_inds],
            train_cat_players[tr_inds]]
    tr_y = [train_y[tr_inds], train_y_raw[tr_inds] / 100]
    val_x = [train_dense_game[val_inds], train_dense_players[val_inds], train_cat_game[val_inds],
             train_cat_players[val_inds]]
    val_y = [train_y[val_inds], train_y_raw[val_inds] / 100]
    model.fit(tr_x,
              tr_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(val_x, val_y))
    loss = model.history.history["val_out_soft_loss"][-1]
    return model, loss


# if __name__ == '__main__':
    # train_dense_game, train_dense_players, train_cat_game, train_cat_players, train_y_raw, train_y, num_classes = test.get_NN_feature()
    # losses = []
    # models = []
    # for k in range(2):
    #     kfold = KFold(5, random_state=42 + k, shuffle=True)
    #     for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(train_y)):
    #         print("-----------")
    #         print("-----------")
    #         model, loss = get_model(train_dense_game, train_dense_players, train_cat_game, train_cat_players,
    #                                 train_y_raw, train_y, num_classes, 32, 20)
    #         models.append(model)
    #         print(k_fold, loss)
    #         losses.append(loss)
    # print("-------")
    # print(losses)
    # print(np.mean(losses))
    # plot_model(model, to_file='model.png')
