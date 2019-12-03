# https://www.kaggle.com/cpmpml/keras-80
# lb .1299

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import datetime
#from kaggle.competitions import nflrush
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import re
import keras
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from keras.utils import Sequence
tqdm.pandas()

from numba import jit

from kaggle.competitions import nflrush

import random as rn
import tensorflow as tf
import numpy as np
from keras import backend as K

def init_seeds(seed):

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res



    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(seed)

    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)
    return sess

input_folder = '../input/nfl-big-data-bowl-2020'
train = pd.read_csv(os.path.join(input_folder, 'train.csv'), dtype={'WindSpeed': 'object'})
train.shape
results = {}

player_cols = ['X', 'Y', 'X_dir', 'Y_dir', 'X_S', 'Y_S', 'S', 'A', 'IsRusher', 'IsOnOffense']
#player_cols = ['X', 'Y', 'X_dir', 'Y_dir', 'X_S', 'Y_S', 'S', 'A', 'Energy', 'IsRusher', 'IsOnOffense']
play_cols = ['X_rusher', 'Y_rusher', 'YardLine_std']


def reorient(train, flip_left):
    train['ToLeft'] = train.PlayDirection == "left"
    # train['IsBallCarrier'] = train.NflId == train.NflIdRusher

    train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    train['TeamOnOffense'] = "home"
    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    train['IsOnOffense'] = train.Team == train.TeamOnOffense  # Is player on offense?
    train['YardLine_std'] = 100 - train.YardLine
    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,
              'YardLine_std'
    ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,
                  'YardLine']
    train.loc[train.ToLeft, 'X'] = 120 - train.loc[train.ToLeft, 'X']
    train.loc[train.ToLeft, 'Y'] = 160 / 3 - train.loc[train.ToLeft, 'Y']
    train.loc[train.ToLeft, 'Orientation'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation'], 360)
    train['Dir'] = 90 - train.Dir
    train.loc[train.ToLeft, 'Dir'] = np.mod(180 + train.loc[train.ToLeft, 'Dir'], 360)
    train.loc[train.IsOnOffense, 'Dir'] = train.loc[train.IsOnOffense, 'Dir'].fillna(0).values
    train.loc[~train.IsOnOffense, 'Dir'] = train.loc[~train.IsOnOffense, 'Dir'].fillna(180).values

    train['IsRusher'] = train['NflId'] == train['NflIdRusher']
    if flip_left:
        df = train[train['IsRusher']].copy()
        # df['left'] = df.Y < 160/6
        df['left'] = df.Dir < 0
        train = train.merge(df[['PlayId', 'left']], how='left', on='PlayId')
        train['Y'] = train.Y
        train.loc[train.left, 'Y'] = 160 / 3 - train.loc[train.left, 'Y']
        train['Dir'] = train.Dir
        train.loc[train.left, 'Dir'] = np.mod(- train.loc[train.left, 'Dir'], 360)
        train.drop('left', axis=1, inplace=True)

    train['X_dir'] = np.cos((np.pi / 180) * train.Dir)
    train['Y_dir'] = np.sin((np.pi / 180) * train.Dir)
    train['X_S'] = train.X_dir * train.S
    train['Y_S'] = train.Y_dir * train.S
    train['X_A'] = train.X_dir * train.A
    train['Y_A'] = train.Y_dir * train.A
    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
    train['BMI'] = (train['PlayerWeight'] * 703) / (train['PlayerHeight'] ** 2)
    train['Energy'] = train['PlayerWeight'] * (train['S'] ** 2)
    train.loc[train['Season'] == 2017, 'S'] = (train['S'][train['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    train['time_step'] = 0.0
    train = train.sort_values(by=['PlayId', 'IsOnOffense', 'IsRusher', 'Y']).reset_index(drop=True)
    return train


def add_features(train):
    df = train[train.IsRusher][['PlayId', 'time_step', 'X', 'Y']].copy()
    df.columns = ['PlayId', 'time_step', 'X_rusher', 'Y_rusher']
    train = train.merge(df, how='left', on=['PlayId', 'time_step'])
    # train.loc[~train.IsRusher, 'X'] = train.loc[~train.IsRusher, 'X'] - train.loc[~train.IsRusher, 'X_rusher']
    # train.loc[~train.IsRusher, 'Y'] = train.loc[~train.IsRusher, 'Y'] - train.loc[~train.IsRusher, 'Y_rusher']
    train.X -= train.X_rusher
    train.Y -= train.Y_rusher

    # train.drop(['Orientation', 'Dir', 'TeamOnOffense', 'YardLine', 'left'], axis=1, inplace=True)
    return train


def time_forward(train, time_step):
    train = train.copy()
    train['X'] = train['X'] + time_step * train['X_S'] + 0.5 * time_step ** 2 * train['X_A']
    train['Y'] = train['Y'] + time_step * train['Y_S'] + 0.5 * time_step ** 2 * train['Y_A']
    x_s = np.clip(train['X_S'] + time_step * train['X_A'], -14, 14)
    y_s = np.clip(train['Y_S'] + time_step * train['Y_A'], -14, 14)
    if 1:
        train['X_A'] = np.clip(train['X_A'] * train['X_S'] / (x_s), -10, 10)
        train['Y_A'] = np.clip(train['Y_A'] * train['Y_S'] / (y_s), -10, 10)
    train['X_S'] = x_s
    train['Y_S'] = y_s
    train['time_step'] = time_step
    return train

train_reorient = reorient(train.copy(), flip_left=True)
len_train_reorient = len(train_reorient) // 22

def augment_data(train_reorient, time_steps):
    train_reorient = pd.concat(time_forward(train_reorient, time_step) for time_step in time_steps)
    train_reorient = add_features(train_reorient)
    return train_reorient

time_steps = [0, 0.3, 0.6]
train_reorient = augment_data(train_reorient, time_steps)

df = train_reorient[player_cols]
df.head(22)

len_2017 = train[train.Season == 2017].shape[0] // 22
len_2017

ss_players = StandardScaler()
X_players = df.values.astype('float32')
X_players = ss_players.fit_transform(df)
X_players = X_players.reshape((-1, 22, len(player_cols)))
X_players.shape

@jit
def get_dmat(X, Y):
    dmat = np.zeros((22, 22))
    for i in range(22):
        for j in range(i+1, 22):
            d = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
            dmat[i, j] = d
            dmat[j, i] = d
    dmat = dmat.reshape((1, 22, 22))
    return dmat

X,Y = train_reorient.X.values, train_reorient.Y.values
dmats = [get_dmat(X[i:i+22], Y[i:i+22]) for i in range(0, train_reorient.shape[0], 22)]
dmats = np.vstack(dmats).reshape((-1, 22, 22))
dmats.shape

inv_dmats = 1. / (1e-2 + dmats)**2
inv_dmats /= inv_dmats.sum(axis=2, keepdims=True)
_ = plt.hist(inv_dmats.ravel(), bins=100, log=True)

X_play = train_reorient[play_cols].values[::22]
ss_play = StandardScaler()
X_play = ss_play.fit_transform(X_play)

del train

from keras import backend as K


class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


y_mae = train_reorient['Yards'].values[::22]
y_mean = np.median(y_mae)
y_crps = np.zeros(shape=(y_mae.shape[0], 199))
for i, yard in enumerate(y_mae):
    y_crps[i, yard + 99:] = 1

yardline = train_reorient['YardLine_std'].values[::22]

def crps(y_true, y_pred):
    stops = np.arange(-99, 100)
    unit_steps = stops >= y_true.reshape(-1, 1)
    crps = np.mean((y_pred - unit_steps)**2)
    return crps

baseline = np.histogram(y_mae, bins = range(-100, 100, 1))[0].cumsum()/len(y_mae)
crps(y_mae, baseline)

def nondecreasing(x):
    X_ir = np.arange(199).astype('float64')
    ir = IsotonicRegression(0, 1)
    x = ir.fit_transform(X_ir, x.astype('float64'))
    return x


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Add, Multiply, Dot
from keras.layers import Embedding, Permute, Reshape, GaussianNoise
from keras.layers.core import Dropout, Lambda, Dense, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.merge import concatenate, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, SGD, Nadam
from keras import backend as K

from keras.engine.topology import Layer
import tensorflow as tf


class ScaleLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, x):
        xx = K.arange(-99, 100, dtype=tf.float32)
        mu = y_mean + tf.reshape(x[:, 0], (-1, 1))
        sigma_minus = tf.identity(K.exp(0.5 * tf.reshape(x[:, 1], (-1, 1))), name="sigma")
        sigma_plus = tf.identity(K.exp(0.5 * tf.reshape(x[:, 2], (-1, 1))), name="sigma")
        xx = tf.subtract(xx, mu)
        pcf = tf.where(xx >= 0, tf.divide(xx, sigma_plus), tf.divide(xx, sigma_minus))
        return pcf

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def message_mult(n, e):
    res = Lambda(lambda c: K.batch_dot(c[0], c[1]), name="mul_1")([e, n])
    res = Lambda(lambda res: K.sum(res, axis=-1))(res)
    res = Lambda(lambda res: K.sum(res, axis=-1, keepdims=True))(res)
    return res


def rusher_dist_mult(dist, players):
    res = Lambda(lambda c: c[0] * c[1])([dist, players])
    return res


def dist_mult(dist, players):
    res = Lambda(lambda c: K.batch_dot(c[0], c[1]))([dist, players])
    return res


def dist_attention(dist, players, dropout):
    if 1:
        dist1 = Reshape((22, 22, 1))(dist)
        dist1 = Conv2D(16, 1, activation='relu',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',
                       )(dist1)
        dist1 = Conv2D(1, 1, activation='relu',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform',
                       )(dist1)
        dist1 = Reshape((22, 22))(dist1)
        dist = Add()([dist, dist1])
    dist = LayerNormalization()(dist)
    att = dist_mult(dist, players, )
    x_player = Add()([players, att])
    x_player = LayerNormalization()(x_player)
    if dropout > 0:
        x_player = Dropout(dropout)(x_player)
    return x_player


def get_rusher(players):
    res = Lambda(lambda c: c[:, -1, :])(players)
    return res


def attention(x, n_factor, dropout):
    x_Q = Conv1D(n_factor, 1, activation='linear',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 )(x)
    x_K = Conv1D(n_factor, 1, activation='linear',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 )(x)
    x_V = Conv1D(n_factor, 1, activation='linear',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 )(x)
    x_KT = Permute((2, 1))(x_K)
    res = Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att


def self_attention(x, n_factor, dropout):
    att = attention(x, n_factor, dropout)
    att = LayerNormalization()(att)
    if dropout > 0:
        att = Dropout(dropout)(att)
    x = Add()([x, att])
    return x


def multi_head_attention(x, n_factor, n_head, dropout):
    n_factor_head = n_factor // n_head
    heads = [attention(x, n_factor_head, dropout) for i in range(n_head)]
    att = Concatenate()(heads)
    att = Dense(n_factor,
                kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform',
                )(att)
    x = Add()([x, att])
    x = LayerNormalization()(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x


def conv_block(players, n_factor, n_hidden, dropout):
    players0 = players
    players = Conv1D(n_hidden, 1, activation='relu',
                     kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform',
                     )(players)
    players = Conv1D(n_factor, 1, activation='relu',
                     kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform',
                     )(players)
    players = Add()([players0, players])
    players = LayerNormalization()(players)
    if dropout > 0:
        players = Dropout(dropout)(players)
    return players


def readout_bloc(readout, n_hidden):
    readout = Dense(3)(readout)
    readout = ScaleLayer(output_dim=199)(readout)
    readout = Reshape((199, 1))(readout)
    readout1 = Conv1D(n_hidden, 1, activation='relu')(readout)
    readout1 = Conv1D(1, 1, activation='relu')(readout1)
    readout = Add()([readout, readout1])
    readout = Flatten()(readout)
    out = keras.layers.Activation('sigmoid')(readout)
    return out


def test_model(n_player, n_factor, n_head):
    input_dmats = Input((n_player, n_player), name="inv_dist")
    dist1 = Reshape((22, 22, 1))(input_dmats)
    out = Conv2D(10, 1)(dist1)
    return Model(inputs=[input_dmats], outputs=[out])


n_player = 22
n_factor = 16
n_loop = 1
n_head = 2
n_player_cols = len(player_cols)
model = test_model(n_player, n_factor, n_head)
model.summary()

def get_model(n_player, n_factor, n_loop, n_head, n_hidden, dropout, n_player_cols, n_play_cols):
    input_players = Input((n_player, n_player_cols), name="players")
    input_dmats = Input((n_player, n_player), name="inv_dist")
    input_play = Input((n_play_cols,), name="plays")

    x_player = input_players
    x_player = Conv1D(n_factor, 1, activation='relu')(x_player)
    x_player = LayerNormalization()(x_player)

    for l in range(n_loop):
        x_player = dist_attention(input_dmats, x_player, dropout)
        x_player = conv_block(x_player, n_factor, n_hidden, dropout)

        x_player = multi_head_attention(x_player, n_factor, n_head, dropout)
        x_player = conv_block(x_player, n_factor, n_hidden, dropout)

    x_play = Dense(n_factor, activation='relu')(input_play)
#    x_play = Dense(n_factor, activation='relu')(x_play)
#    x_play = LayerNormalization()(x_play)
#    x_play = Dropout(dropout)(x_play)
    x_rusher = get_rusher(x_player)
    readout = Concatenate()([x_rusher, x_play])
    #readout = Dense(n_factor, activation='relu')(readout)
    #readout = Dropout(dropout)(readout)
    out1 = Dense(199, activation='sigmoid')(readout)
    if 1:
        readout = Dense(3)(readout)
        readout = ScaleLayer(output_dim=199)(readout)
        out2 = keras.layers.Activation('sigmoid')(readout)
    else:
        out2 = readout_bloc(readout, n_factor)
    return Model(inputs=[input_players, input_dmats, input_play], outputs=[out1, out2])

n_player = 22
n_factor = 16
n_hidden = 2*n_factor
n_loop = 1
n_head = 2
dropout = 0.1
n_player_cols = len(player_cols)
n_play_cols = len(play_cols)
model = get_model(n_player, n_factor, n_loop, n_head, n_hidden, dropout, n_player_cols, n_play_cols)
model.summary()

simulate_test = False

if simulate_test:
    len_data = X_players.shape[0]
    len_train = int(0.9 * len_data)
else:
    len_data = X_players.shape[0]
    len_train = len_data

X_players_train, X_players_test = X_players[:len_train], X_players[len_train:]
inv_dmats_train, inv_dmats_test = inv_dmats[:len_train], inv_dmats[len_train:]
X_play_train, X_play_test = X_play[:len_train], X_play[len_train:]
y_mae_train, y_mae_test = y_mae[:len_train], y_mae[len_train:]
y_crps_train, y_crps_test = y_crps[:len_train], y_crps[len_train:]
yardline_train, yardline_test = yardline[:len_train], yardline[len_train:]

kf = KFold(n_splits=12, shuffle=False)
ind_2017 = np.arange(len_2017)
def add_2017(x,y):
    x = np.concatenate((ind_2017, x +  len_2017))
    y = y + len_2017
    return x,y

indices = [add_2017(x, y) for x, y in kf.split(X_players_train[len_2017:len_train_reorient],
                                               y_mae_train[len_2017:len_train_reorient])]

n_steps = len(time_steps)

def augment_indices(x, y, time_steps):
    x_a = np.concatenate([x + i*len_train_reorient for i in range(n_steps)])
    y_a = np.concatenate([y + i*len_train_reorient for i in range(n_steps)])
    return x_a, y_a

indices = [augment_indices(x, y, time_steps) for x,y in indices ]
indices = [(i, ind) for i,ind in enumerate(indices)]

w = np.ones(len_train_reorient)
w[:len_2017] = 0.5
w = np.concatenate([w for i in range(n_steps)])

def post_process(y_pred, y_mae_train, yardline_train, time_steps, score=True):
    if score:
        score0 = crps(y_mae_train, y_pred)
    else:
        score0 = 0
    upper = 100 - yardline_train
    lower = - yardline_train
    for i in range(y_pred.shape[0]):
        y_pred[i, 99 + upper[i]:] = 1
        y_pred[i, :99 + lower[i]] = 0
        y_pred[i] = nondecreasing(y_pred[i].ravel())
    if score:
        score1 = crps(y_mae_train, y_pred)
    else:
        score1 = 0
    if 1:
        n_steps = len(time_steps)
        len_y_pred = len(y_pred) // n_steps
        y_pred1 = np.mean([y_pred[len_y_pred * i: len_y_pred * (i+1)] for i in range(n_steps)], axis=0)
        score2 = crps(y_mae_train[:len_y_pred], y_pred1)
    return y_pred1, score0, score1, score2

def post_process(y_pred, yardline_train, time_steps):
    upper = 100 - yardline_train
    lower = - yardline_train
    for i in range(y_pred.shape[0]):
        y_pred[i, 99 + upper[i]:] = 1
        y_pred[i, :99 + lower[i]] = 0
        y_pred[i] = nondecreasing(y_pred[i].ravel())
    if 1:
        n_steps = len(time_steps)
        len_y_pred = len(y_pred) // n_steps
        y_pred = np.mean([y_pred[len_y_pred * i: len_y_pred * (i+1)] for i in range(n_steps)], axis=0)
    return y_pred


name = 'nn'
results[name] = {'crps': [], 'y_test': [], 'imp': [], 'models': []}

try:
    del sess
except:
    pass

sess = init_seeds(0)

pred_test = []

for index in indices:
    if 1 and index[0] < 10:
        continue
    train_idx, val_idx = index[1]
    train_players, train_inv_dmats, train_play, train_y, = (X_players_train[train_idx],
                                                            inv_dmats_train[train_idx],
                                                            X_play_train[train_idx],
                                                            y_crps_train[train_idx],)
    val_players, val_inv_dmats, val_play, val_y, = (X_players_train[val_idx],
                                                    inv_dmats_train[val_idx],
                                                    X_play_train[val_idx],
                                                    y_crps_train[val_idx],)
    w_train = w[train_idx]

    n_player = 22
    n_factor = 64
    n_loop = 1
    n_head = 4
    n_hidden = 2 * n_factor
    dropout = 0.25
    n_player_cols = len(player_cols)
    model = get_model(n_player, n_factor, n_loop, n_head, n_hidden, dropout, n_player_cols, n_play_cols)
    # model.summary()

    if 1:
        opm = keras.optimizers.Adam(lr=1e-3)
        # opm = RAdam(warmup_proportion=0.1, min_lr=1e-5)
        model.compile(loss='mse', optimizer=opm, metrics=[])
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                           restore_best_weights=True, verbose=0, patience=21)
        lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='min',
                                               min_delta=0.00001)
        history = model.fit([train_players, train_inv_dmats, train_play], [train_y, train_y],
                            verbose=2, batch_size=256, epochs=500,
                            callbacks=[es, lr],
                            validation_data=[[val_players, val_inv_dmats, val_play], [val_y, val_y]],
                            sample_weight=[w_train, w_train])

    X_ir = np.arange(199).astype('float64')
    # print(val_players.shape, val_inv_dmats.shape)

    y_pred = model.predict([val_players, val_inv_dmats, val_play])
    y_pred = (y_pred[0] + y_pred[1]) / 2.
    score0 = crps(y_mae_train[val_idx], y_pred)
    y_pred = post_process(y_pred, yardline_train[val_idx], time_steps)
    score1 = crps(y_mae_train[val_idx][:len(y_pred)], y_pred)
    print(' val fold:', index[0], score0)
    print(' val fold:', index[0], score1)
    if score0 < score1:
        break
    results[name]['models'].append(model)
    results[name]['crps'].append(score1)
    # results[name]['imp'].append(model.feature_importances_)
    if simulate_test:
        y_pred = model.predict([X_players_test, inv_dmats_test, X_play_test])
        y_pred = (y_pred[0] + y_pred[1]) / 2.
        score0 = crps(y_mae_test, y_pred)
        y_pred = post_process(y_pred, yardline_test, time_steps)
        score1 = crps(y_mae_test[:len(y_pred)], y_pred)
        print('test fold:', index[0], score0)
        print('test fold:', index[0], score1)
        pred_test.append(y_pred)
    print('*' * 40)
print('Folds:', results[name]['crps'])
print('Average:', np.mean(results[name]['crps']))

Folds = [0.0134185115, 0.012763181]
Average = 0.013090846

from tqdm import tqdm

env = nflrush.make_env()
iter_test = env.iter_test()
del train_reorient
for (test_df, sample_prediction_df) in tqdm(iter_test):
    test_reorient = reorient(test_df.copy(), flip_left=True)
    test_reorient = augment_data(test_reorient, time_steps)
    df = test_reorient[player_cols]
    X_players = df.values.astype('float32')
    X_players = ss_players.transform(df)
    X_players = X_players.reshape((-1, 22, len(player_cols)))
    X,Y = test_reorient.X.values, test_reorient.Y.values
    dmats = [get_dmat(X[i:i+22], Y[i:i+22]) for i in range(0, test_reorient.shape[0], 22)]
    dmats = np.vstack(dmats).reshape((-1, 22, 22))
    inv_dmats = 1. / (1e-2 + dmats)**2
    inv_dmats /= inv_dmats.sum(axis=2, keepdims=True)
    X_play = test_reorient[play_cols].values[::22]
    X_play = ss_play.transform(X_play)
    yardline = test_reorient['YardLine_std'].values[::22]
    y_preds = [model.predict([X_players, inv_dmats, X_play]
                            ) for model in results[name]['models']]
    y_preds = [(pred[0] + pred[1]) / 2. for pred in y_preds]
    y_pred = np.mean(y_preds, axis=0)
    y_pred = post_process(y_pred, yardline, time_steps)
    sample_prediction_df.iloc[0, :] = y_pred.ravel()
    env.predict(sample_prediction_df)
env.write_submission_file()

X,Y = test_reorient.X.values, test_reorient.Y.values
dmats = [get_dmat(X[i:i+22], Y[i:i+22]) for i in range(0, test_reorient.shape[0], 22)]
dmats = np.vstack(dmats).reshape((-1, 22, 22))
dmats.shape