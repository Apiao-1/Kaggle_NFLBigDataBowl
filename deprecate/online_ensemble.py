# https://www.kaggle.com/enzoamp/nfl-lightgbm/code
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 150)


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


def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x / 15))
    except:
        return "nan"


def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_Y(y_coordinate, play_direction):
        if play_direction == 'left':
            return 160.0 / 3 - y_coordinate
        else:
            return y_coordinate

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

    # def new_orientation(angle, play_direction):
    #     if play_direction == 'left':
    #         new_angle = (180.0 + angle) % 360
    #         return new_angle
    #     else:
    #         return angle

    def euclidean_distance(x1, y1, x2, y2):
        x_diff = (x1 - x2) ** 2
        y_diff = (y1 - y2) ** 2

        return np.sqrt(x_diff + y_diff)

    def q80(x):
        return x.quantile(0.8)

    def q30(x):
        return x.quantile(0.3)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam', 'FieldPosition', 'YardLine']].apply(
            lambda x: new_line(x[0], x[1], x[2]), axis=1)
        new_yardline['Yards_limit'] = 110 - new_yardline['YardLine']
        new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine', 'Yards_limit']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X', 'PlayDirection']].apply(lambda x: new_X(x[0], x[1]), axis=1)
        df['Y'] = df[['Y', 'PlayDirection']].apply(lambda x: new_Y(x[0], x[1]), axis=1)
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
            ['GameId', 'PlayId', 'NflIdRusher', 'back_X', 'back_Y', 'back_from_scrimmage'
                , 'back_oriented_down_field', 'back_moving_down_field'
             ]]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId', 'PlayId', 'NflId', 'X', 'Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X', 'Y', 'back_X', 'back_Y']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        player_distance = player_distance.groupby(
            ['GameId', 'PlayId', 'back_from_scrimmage',
             'back_oriented_down_field', 'back_moving_down_field'
             ]) \
            .agg({'dist_to_back': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad',
                                   np.ptp],
                  'X': ['mean', 'std'],
                  }) \
            .reset_index()

        player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage',
                                   'back_oriented_down_field', 'back_moving_down_field',
                                   'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'skew_dist', 'medn_dist',
                                   'q80_dist', 'q30_dist', 'kurt_dist', 'mad_dist', 'ptp_dist',
                                   'X_mean', 'X_std', ]

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY', 'RusherS', 'RusherA']

        defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        defense_d = defense[defense['Team'] != defense['RusherTeam']][
            ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        defense_d['def_dist_to_back'] = defense_d[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        defense_d = defense_d.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt,
                                       'mad', np.ptp],
                  'X': ['mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  }) \
            .reset_index()
        defense_d.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist',
                             'def_skew_dist', 'def_medn_dist', 'def_q80_dist', 'def_q30_dist', 'def_kurt_dist',
                             'def_mad_dist', 'def_ptp_dist',
                             'def_X_mean', 'def_X_std', 'def_X_skew', 'def_X_median', 'def_X_q80', 'def_X_q30',
                             'def_X_kurt', 'def_X_mad', 'def_X_ptp']

        defense_s = defense[defense['Team'] != defense['RusherTeam']][['GameId', 'PlayId', 'S', 'A']]
        defense_s['SA'] = defense_s[['S', 'A']].apply(lambda x: x[0] + x[1], axis=1)
        defense_s = defense_s.groupby(['GameId', 'PlayId']) \
            .agg({'S': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'A': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'SA': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  }) \
            .reset_index()
        defense_s.columns = ['GameId', 'PlayId', 'def_min_s', 'def_max_s', 'def_mean_s', 'def_std_s', 'def_skew_s',
                             'def_medn_s', 'def_q80_s', 'def_q30_s', 'def_kurt_s', 'def_mad_s', 'def_ptp_s',
                             'def_min_a', 'def_max_a', 'def_mean_a', 'def_std_a', 'def_skew_a', 'def_medn_a',
                             'def_q80_a', 'def_q30_a', 'def_kurt_a', 'def_mad_a', 'def_ptp_a', 'def_min_sa',
                             'def_max_sa', 'def_mean_sa', 'def_std_sa', 'def_skew_sa', 'def_medn_sa', 'def_q80_sa',
                             'def_q30_sa', 'def_kurt_sa', 'def_mad_sa', 'def_ptp_sa', ]

        defense = pd.merge(defense_d, defense_s, on=['GameId', 'PlayId'], how='inner')

        return defense

    def team_features(df):
        # centroid dis, abs statistic
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY', 'RusherS', 'RusherA']

        team = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        team_d = team[team['Team'] == team['RusherTeam']][['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        team_d['def_dist_to_back'] = team_d[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        team_d = team_d.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back': [
            'min',
            'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt,
            'mad', np.ptp]}) \
            .reset_index()
        team_d.columns = ['GameId', 'PlayId',
                          'tm_min_dist',
                          'tm_max_dist', 'tm_mean_dist', 'tm_std_dist',
                          'tm_skew_dist', 'tm_medn_dist', 'tm_q80_dist', 'tm_q30_dist', 'tm_kurt_dist', 'tm_mad_dist',
                          'tm_ptp_dist']

        team_s = team[team['Team'] == team['RusherTeam']][['GameId', 'PlayId', 'S', 'A']]
        team_s['SA'] = team_s[['S', 'A']].apply(lambda x: x[0] + x[1], axis=1)
        team_s = team_s.groupby(['GameId', 'PlayId']) \
            .agg({'S': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'A': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'SA': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp]}) \
            .reset_index()
        team_s.columns = ['GameId', 'PlayId', 'tm_min_s', 'tm_max_s', 'tm_mean_s', 'tm_std_s', 'tm_skew_s', 'tm_medn_s',
                          'tm_q80_s', 'tm_q30_s', 'tm_kurt_s', 'tm_mad_s', 'tm_ptp_s', 'tm_min_a', 'tm_max_a',
                          'tm_mean_a', 'tm_std_a', 'tm_skew_a', 'tm_medn_a', 'tm_q80_a', 'tm_q30_a', 'tm_kurt_a',
                          'tm_mad_a', 'tm_ptp_a', 'tm_min_sa', 'tm_max_sa', 'tm_mean_sa', 'tm_std_sa', 'tm_skew_sa',
                          'tm_medn_sa', 'tm_q80_sa', 'tm_q30_sa', 'tm_kurt_sa', 'tm_mad_sa', 'tm_ptp_sa']

        team = pd.merge(team_d, team_s, on=['GameId', 'PlayId'], how='inner')

        return team

    # tested
    def static_features(df):

        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        # df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60 * 60 * 24 * 365.25
        df['PlayerAge'] = df.apply(
            lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party','partly') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(
            lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')

        df["Dir_ob"] = df["Dir"].apply(lambda x: orientation_to_cat(x)).astype("object")
        add_new_feas.append("Dir_ob")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")
        add_new_feas.append("Orientation_cos")
        add_new_feas.append("Orientation_sin")

        ## Turf
        grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
        df['Grass'] = np.where(df.Turf.str.lower().isin(grass_labels), 1, 0)
        add_new_feas.append("Grass")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")

        static_features = df[df['NflId'] == df['NflIdRusher']][
            add_new_feas + ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',
                            'YardLine', 'Yards_limit', 'Quarter', 'Down', 'Distance',
                            'DefendersInTheBox']].drop_duplicates()

        static_features.fillna(0, inplace=True)

        return static_features

    def combine_features(relative_to_back, defense, team, static, deploy=deploy):
        df = pd.merge(relative_to_back, defense, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, team, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, static, on=['GameId', 'PlayId'], how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId', 'PlayId'], how='inner')

        return df

    # if deploy == False:
    #     df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[train['Season'] == 2017, 'Orientation'], 360)
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    tm_feats = team_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, tm_feats, static_feats, deploy=deploy)

    # print(df.shape, back_feats.shape, rel_back.shape, def_feats.shape, static_feats.shape, basetable.shape)

    return basetable

# tested
def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) + np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    # t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    t_["nextS"] = t_["S"] + t_["A"]
    t_["Sv"] = t_["S"] * np.cos(radian_angle)
    # t_["Sh"] = t_["S"] * np.sin(radian_angle)
    t_["Av"] = t_["A"] * np.cos(radian_angle)
    # t_["Ah"] = t_["A"] * np.sin(radian_angle)
    t_["diff_ang"] = t_["Dir"] - t_["Orientation"]
    return t_


class CRPSCallback(Callback):

    def __init__(self, validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    #         print('validation shape',len(self.validation))

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
            val_s = np.round(val_s, 8)
            logs['CRPS_score_val'] = val_s


def get_model(x_tr, y_tr, x_val, y_val):
    inp = Input(shape=(x_tr.shape[1],))
    # x = Dense(2048, input_dim=X.shape[1], activation='elu')(inp)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)
    # x = Dense(1024, activation='elu')(x)
    x = Dense(1024, input_dim=X.shape[1], activation='elu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    if classify_type < 128:
        x = Dense(128, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
    # if classify_type < 64:
    #     x = Dense(64, activation='elu')(x)
    #     x = Dropout(0.5)(x)
    #     x = BatchNormalization()(x)

    out = Dense(classify_type, activation='softmax')(x)
    model = Model(inp, out)
    optadam = Adam(lr=0.001)
    model.compile(optimizer=optadam, loss='categorical_crossentropy', metrics=[])

    es = EarlyStopping(monitor='CRPS_score_val',
                       mode='min',
                       restore_best_weights=True,
                       verbose=False,
                       patience=80)

    mc = ModelCheckpoint('best_model.h5', monitor='CRPS_score_val', mode='min',
                         save_best_only=True, verbose=False, save_weights_only=True)

    bsz = 1024
    steps = x_tr.shape[0] / bsz

    model.fit(x_tr, y_tr, callbacks=[CRPSCallback(validation=(x_val, y_val)), es, mc], epochs=100, batch_size=bsz,
              verbose=False)
    model.load_weights("best_model.h5")

    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)

    # x_val['Yards_limit']

    # y_0 = np.zeros((len(y_pred), 85))
    # y_pred = np.concatenate((y_pred, y_0), axis=1)
    # print(y_pred.shape)

    #     y_pred[y_pred<0.01] = 0.0
    #     y_pred[y_pred>0.99] = 1.0
    #     for index,item in enumerate(y_pred):
    #         if item<0.001:
    #             y_pred[index] = 0.0
    #         elif item>0.999:
    #             y_pred[index] = 1.0
    #         else:
    #             y_pred[index] = y_pred[index]
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 8)

    return model, crps


def predict(x_te):
    model_num = len(models)
    for k, m in enumerate(models):
        if k == 0:
            y_pred = m.predict(x_te, batch_size=1024)
        else:
            y_pred += m.predict(x_te, batch_size=1024)

    y_pred = y_pred / model_num

    return y_pred




def train_lgb(X, y, single=True):
    lgb_models = []
    scores = []
    kf = KFold(n_splits=5, random_state=42)
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
        model = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data],verbose_eval=False,
                          early_stopping_rounds=60)
        lgb_models.append(model)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        # print(y_pred.shape)  # (200, 199)
        score_ = metric_crps(y_true, y_pred)
        # print(score_)
        scores.append(score_)
        lgb_models.append(model)
    print("lgb mean score:", np.mean(scores))

    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return lgb_models[min_index]
        return lgb_models[0]
    return lgb_models

def train_rf(X, y, single=True):
    rf_models = []
    scores = []
    kf = KFold(n_splits=5, random_state=42)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        # print(X_train.shape, y_train.shape)  # (800, 199)
        # print(X_val.shape, y_val.shape)
        model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=15, min_samples_split=7,
                                      n_estimators=250, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # print(y_pred.shape) # (200, 199)
        score_ = metric_crps(y_val, y_pred)
        # print(score_)
        scores.append(score_)
        rf_models.append(model)
    print("rf mean score:", np.mean(scores))

    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return rf_models[min_index]
        return rf_models[0]
    return rf_models

def train_NN(X, y, single=True, seed = 42):
    models = []
    scores = []
    kf = KFold(n_splits=5, random_state=seed)
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        # print(X_train.shape, y_train.shape)  # (800, 199)
        # print(X_val.shape, y_val.shape)
        model, crps = get_NN_model(X_train, y_train, X_val, y_val)
        models.append(model)
        scores.append(crps)
    print("NN mean score:", np.mean(scores))

    if single:
        # min_index = 0
        # min = scores[min_index]
        # for i, s in scores:
        #     if s < min:
        #         min = s
        #         min_index = i
        # return models[min_index]
        return models[0]
    return models


TRAIN_OFFLINE = False
CLASSIFY_NEGITAVE = -14  # must < 0
CLASSIFY_POSTIVE = 99  # 99， 75，53， 36
classify_type = CLASSIFY_POSTIVE - CLASSIFY_NEGITAVE + 1

if __name__ == '__main__':
    path = '/Users/a_piao/PycharmProjects/my_competition/NFLBigDataBowl/cache_feature.csv'
    if TRAIN_OFFLINE:
        if os.path.exists(path):
            train_basetable = pd.read_csv(path)
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

    clfs = ['NN', 'rf', 'lgb']
    models = []
    for j, v in enumerate(clfs):
        clf = eval('train_%s' % v)(X, y, False)
        models.append(clf)
    models.append(train_NN(X,y,False, seed=43))
    # flatten
    # models = sum(models, [])
    # print(len(models))

    for model_list in models:
        y_pred = np.mean([model.predict(scaled_basetable) for model in model_list], axis=0)

    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        basetable = create_features(test_df, deploy=True)

        basetable.drop(['GameId', 'PlayId'], axis=1, inplace=True)
        scaled_basetable = scaler.transform(basetable)

        y_pred = np.mean([model.predict(scaled_basetable) for model in models], axis=0)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
        preds_df.iloc[:, :50] = 0
        preds_df.iloc[:, -50:] = 1
        env.predict(preds_df)

    env.write_submission_file()
