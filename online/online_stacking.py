# https://www.kaggle.com/enzoamp/nfl-lightgbm/code
# 这个版本的stacking不再试用，原因：预测出的概率不是严格递增的
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import datetime
import warnings
import os
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda
from keras.models import Model
from keras.layers import BatchNormalization
import keras.backend as K
import re

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 150)


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def early_metric_crps(y_true, y_pred):
    # print(y_pred.shape, y_true.shape)
    y = np.zeros((y_true.shape[0], 199))
    for idx, target in enumerate(list(y_true)):
        y[idx][99 + target] = 1
    y_true = np.clip(np.cumsum(y, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    crps = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])
    # print(crps)
    return "CRPS", crps


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0]) * 60 + int(txt[1]) + int(txt[2]) / 60
    return ans


def strtofloat(x):
    try:
        return float(x)
    except:
        return -1


def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans *= 0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans * 3
    if 'sunny' in txt or 'sun' in txt:
        return ans * 2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2 * ans
    if 'snow' in txt:
        return -3 * ans
    return 0


def OffensePersonnelSplit(x):
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0, 'QB': 0, 'RB': 0, 'TE': 0, 'WR': 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic


def DefensePersonnelSplit(x):
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic


def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x / 15))
    except:
        return "nan"


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    return txt


def transform_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0

    return np.nan


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1, y1, x2, y2):
        x_diff = (x1 - x2) ** 2
        y_diff = (y1 - y2) ** 2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam', 'FieldPosition', 'YardLine']].apply(
            lambda x: new_line(x[0], x[1], x[2]), axis=1)
        new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X', 'PlayDirection']].apply(lambda x: new_X(x[0], x[1]), axis=1)
        df['Orientation'] = df[['Orientation', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]), axis=1)
        df['Dir'] = df[['Dir', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId', 'PlayId'], how='inner')

        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][
            ['GameId', 'PlayId', 'NflIdRusher', 'X', 'Y', 'Orientation', 'Dir', 'YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X': 'back_X',
                                            'Y': 'back_Y'})
        carriers = carriers[
            ['GameId', 'PlayId', 'NflIdRusher', 'back_X', 'back_Y', 'back_from_scrimmage', 'back_oriented_down_field',
             'back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId', 'PlayId', 'NflId', 'X', 'Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X', 'Y', 'back_X', 'back_Y']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        player_distance = player_distance.groupby(
            ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field', 'back_moving_down_field']) \
            .agg({'dist_to_back': ['min', 'max', 'mean', 'std']}) \
            .reset_index()
        player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field',
                                   'back_moving_down_field',
                                   'min_dist', 'max_dist', 'mean_dist', 'std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY']

        defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][
            ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        defense['def_dist_to_back'] = defense[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        defense = defense.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back': ['min', 'max', 'mean', 'std']}) \
            .reset_index()
        defense.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist']

        return defense

    def static_features(df):

        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60 * 60 * 24 * 365.25
        df['PlayerAge'] = df.apply(
            lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        # df['WindSpeed_ob'] = df['WindSpeed'].apply(
        #     lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        # df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(
        #     lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)
        # df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(
        #     lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(x) and type(
        #         x) != float and 'gusts up to' in x else x)
        # df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        # add_new_feas.append('WindSpeed_dense')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party',
                                                                                        'partly') if not pd.isna(
                x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')
        #         ## Rusher
        #         train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
        #         train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
        #         temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
        #         train = train.merge(temp, on = "PlayId")
        #         train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

        ## dense -> categorical
        #         train["Quarter_ob"] = train["Quarter"].astype("object")
        #         train["Down_ob"] = train["Down"].astype("object")
        #         train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
        #         train["YardLine_ob"] = train["YardLine"].astype("object")
        # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
        # train["Week_ob"] = train["Week"].astype("object")
        # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")

        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x: orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x: orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        add_new_feas.append("Orientation_cos")
        add_new_feas.append("Orientation_sin")
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")

        ## Turf
        grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
        df['Grass'] = np.where(df.Turf.str.lower().isin(grass_labels), 1, 0)
        add_new_feas.append("Grass")

        # StadiumType
        # df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
        # df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)
        # add_new_feas.append("StadiumType")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")

        static_features = df[df['NflId'] == df['NflIdRusher']][
            add_new_feas + ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',
                            'YardLine', 'Quarter', 'Down', 'Distance', 'DefendersInTheBox']].drop_duplicates()
        #         static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))
        static_features.fillna(-999, inplace=True)
        #         for i in add_new_feas:
        #             static_features[i] = static_features[i].fillna(np.mean(static_features[i]))

        return static_features

    def combine_features(relative_to_back, defense, static, deploy=deploy):
        df = pd.merge(relative_to_back, defense, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, static, on=['GameId', 'PlayId'], how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId', 'PlayId'], how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, deploy=deploy)

    return basetable


class CRPSCallback(Callback):

    def __init__(self, validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

        # print('validation shape', len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')

        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


def get_NN_model(x_tr, y_tr, x_val, y_val):
    inp = Input(shape=(x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    out = Dense(199, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    # add lookahead
    #     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
    #     lookahead.inject(model) # add into model

    es = EarlyStopping(monitor='CRPS_score_val',
                       mode='min',
                       # restore_best_weights=True,
                       verbose=0,
                       patience=10)

    mc = ModelCheckpoint('best_model.h5', monitor='CRPS_score_val', mode='min',
                         save_best_only=True, verbose=0, save_weights_only=True)

    bsz = 1024
    steps = x_tr.shape[0] / bsz

    # for i in range(1):
    #     model.fit(x_tr, y_tr, batch_size=32, verbose=False)
    # for i in range(1):
    #     model.fit(x_tr, y_tr, batch_size=64, verbose=False)
    # for i in range(1):
    #     model.fit(x_tr, y_tr, batch_size=128, verbose=False)
    # for i in range(1):
    #     model.fit(x_tr, y_tr, batch_size=256, verbose=False)
    model.fit(x_tr, y_tr, callbacks=[CRPSCallback(validation=(x_val, y_val)), es, mc], epochs=100, batch_size=bsz,
              verbose=0)
    # model.load_weights("best_model.h5")

    return model


def train_lgb(X, y, n_splits, single=False):
    lgb_models = []
    scores = []
    predict_test = np.zeros((y.shape[0], y.shape[1]))
    kf = KFold(n_splits=n_splits, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        # print(f'Fold : {i}')
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        y_true = y_val.copy()

        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)
        # print(X_train.shape, y_train.shape)  # (800, 199)
        # print(X_val.shape, y_val.shape)  # (800, 199)

        param = {
            # 'n_estimators': 500,
            'learning_rate': 0.005,

            'num_leaves': 8,  # Original 50
            'max_depth': 3,

            'min_data_in_leaf': 101,  # min_child_samples
            'max_bin': 75,
            'min_child_weight': 1,

            "feature_fraction": 0.8,  # 0.9 colsample_bytree
            "bagging_freq": 1,
            "bagging_fraction": 0.8,  # 'subsample'
            "bagging_seed": 42,

            'min_split_gain': 0,
            "lambda_l1": 0.01,
            "lambda_l2": 0.01,

            "boosting": "gbdt",
            'num_class': 199,  # 199 possible places
            'objective': 'multiclass',
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 42,
        }

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        num_round = 10000
        model = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=False,
                          early_stopping_rounds=60)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        predict_test[vdx] = y_pred
        # print(y_pred.shape)  # (200, 199)
        score_ = metric_crps(y_true, y_pred)
        # print(score_)
        scores.append(score_)
        lgb_models.append(model)
    print("lgb mean score:", np.mean(scores))
    predict_test = np.clip(np.cumsum(predict_test, axis=1), 0, 1)

    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return lgb_models[min_index]
        return lgb_models[0]
    return lgb_models, predict_test


def train_rf(X, y, n_splits, single=False):
    rf_models = []
    scores = []
    predict_test = np.zeros((y.shape[0], y.shape[1]))
    kf = KFold(n_splits=n_splits, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        # print(X_train.shape, y_train.shape)  # (800, 199)
        # print(X_val.shape, y_val.shape)
        model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=15, min_samples_split=7,
                                      n_estimators=250, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        predict_test[vdx] = y_pred
        # print(y_pred.shape) # (200, 199)
        score_ = metric_crps(y_val, y_pred)
        # print(score_)
        scores.append(score_)
        rf_models.append(model)
    print("rf mean score:", np.mean(scores))
    predict_test = np.clip(np.cumsum(predict_test, axis=1), 0, 1)
    # print(predict_test.shape) # (2200, 199)
    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return rf_models[min_index]
        return rf_models[0]
    return rf_models, predict_test


def train_NN(X, y, n_splits, single=False, seed=42):
    models = []
    scores = []
    predict_test = np.zeros((y.shape[0], y.shape[1]))
    kf = KFold(n_splits=n_splits, random_state=seed)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        # print(X_train.shape, y_train.shape)  # (800, 199)
        # print(X_val.shape, y_val.shape)
        model = get_NN_model(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val)
        predict_test[vdx] = y_pred
        crps = metric_crps(y_val, y_pred)
        # crps = np.round(val_s, 7)

        models.append(model)
        scores.append(crps)
    print("NN mean score:", np.mean(scores))
    predict_test = np.clip(np.cumsum(predict_test, axis=1), 0, 1)

    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return models[min_index]
        return models[0]
    return models, predict_test


TRAIN_OFFLINE = False

if __name__ == '__main__':
    path = '/Users/a_piao/PycharmProjects/my_competition/NFLBigDataBowl/cache_feature.csv'
    if TRAIN_OFFLINE:
        if os.path.exists(path):
            train_basetable = pd.read_csv(path)[:1000]
        else:
            train = pd.read_csv('../data/train.csv', dtype={'WindSpeed': 'object'})
            outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()
            train_basetable = create_features(train, False)
            train_basetable.to_csv(path, index=False)
    else:
        train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
        outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()
        train_basetable = create_features(train, False)

    X = train_basetable.copy()
    yards = X.Yards
    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1
    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = []
    all_test = []
    clfs = ['rf', 'lgb', 'NN']
    for j, v in enumerate(clfs):
        clf, single = eval('train_%s' % v)(X, y, 5, False)
        models.append(clf)
        all_test.append(single)
    print(len(all_test), len(all_test[0]))
    print(len(models), len(models[0]))

    print('blending')
    clf_map = {}

    for idx, target in enumerate(list(yards)):
        y[idx][99 + target:] = 1

    clf_beg = 50
    clf_end = 150
    for idx in range(clf_beg, clf_end):
        single_flag = True
        global single_train
        for v in all_test:
            if single_flag:
                single_train = v[:, idx].reshape(-1, 1)
                # print(single_train.shape)
                single_flag = False
            else:
                single_train = np.concatenate((single_train, v[:, idx].reshape(-1, 1)), axis=1)
                # print(single_train.shape)
        # print(single_train.shape)
        single_clf = SVR()
        # single_clf = Ridge()
        single_clf.fit(single_train, y[:, idx])
        clf_map[idx] = single_clf
        # print(clf_map)

    # flatten
    # models = sum(models, [])
    # print(len(models))

    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        basetable = create_features(test_df, deploy=True)

        basetable.drop(['GameId', 'PlayId'], axis=1, inplace=True)
        scaled_basetable = scaler.transform(basetable)

        y_conduct = []
        for model_list in models:
            y_pred = np.mean([model.predict(scaled_basetable) for model in model_list], axis=0)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            y_conduct.append(y_pred)
            # if flag:
            #     y_conduct = y_pred
            #     flag = False
            # else:
            #     y_conduct = np.concatenate((y_conduct, y_pred))

        #         print(len(y_conduct))
        # y_pred = clf.predict_proba(y_conduct)
        y_ans = np.zeros((1, 199))

        for idx in range(clf_beg, clf_end):
            single_flag = True
            global single_test
            for v in y_conduct:
                if single_flag:
                    single_test = v[:, idx].reshape(-1, 1)
                    single_flag = False
                else:
                    single_test = np.concatenate((single_test, v[:, idx].reshape(-1, 1)), axis=1)
            single_pred = clf_map[idx].predict(single_test)
            y_ans[:, idx] = single_pred[0]

        y_ans[clf_end:] = 1
        y_ans = np.clip(y_ans, 0, 1)
        preds_df = pd.DataFrame(data=y_ans, columns=sample_prediction_df.columns)
        preds_df.iloc[:, :50] = 0
        preds_df.iloc[:, -50:] = 1
        env.predict(preds_df)
    env.write_submission_file()
