# https://www.kaggle.com/enzoamp/nfl-lightgbm/code
import sys
sys.path.append('/home/aistudio/external-libraries')
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import datetime
import warnings
import os

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 50)
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
            lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party',
                                                                                        'partly') if not pd.isna(
                x) else x)
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

    # logging.info(df.shape, back_feats.shape, rel_back.shape, def_feats.shape, static_feats.shape, basetable.shape)

    return basetable


def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) + np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    t_["nextS"] = t_["S"] + t_["A"]
    t_["Sv"] = t_["S"] * np.cos(radian_angle)
    t_["Sh"] = t_["S"] * np.sin(radian_angle)
    t_["Av"] = t_["A"] * np.cos(radian_angle)
    t_["Ah"] = t_["A"] * np.sin(radian_angle)
    t_["diff_ang"] = t_["Dir"] - t_["Orientation"]
    return t_


best_score = 9999
best_param = {}


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def crps_eval(y_pred, dataset, is_higher_better=False):
    labels = dataset.get_label()
    y_true = np.zeros((len(labels), classify_type))
    for i, v in enumerate(labels):
        y_true[i, int(v):] = 1
    y_pred = y_pred.reshape(-1, classify_type, order='F')
    y_pred = np.clip(y_pred.cumsum(axis=1), 0, 1)
    return 'crps', np.mean((y_pred - y_true) ** 2), False


def find_best_param(X, y, params):
    kf = KFold(n_splits=5, random_state=2019)
    score = []
    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        y_true = y_val.copy()
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        num_round = 10000
        model = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=False,
                          early_stopping_rounds=150, feval=crps_eval)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        score_ = metric_crps(y_true, y_pred)
        # logging.info("%d folder with score %f" % (i, score_))
        score.append(score_)
    mean_score = np.mean(score)
    logging.info("mean_score: %f" % mean_score)
    global best_score, best_param
    if mean_score <= best_score:
        best_score = mean_score
        logging.info("update best_score: %f" % best_score)
        best_param = params
        logging.info("update best params: %s" % best_param)


import logging
from logging.handlers import TimedRotatingFileHandler
import re

def init_log():
    logging.getLogger('bloomfilter').setLevel('WARN')
    log_file_handler = TimedRotatingFileHandler(filename="bloomfilter.log", when="D", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s- %(filename)s:%(lineno)s - %(threadName)s - %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler.setFormatter(formatter)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(log_file_handler)

TRAIN_OFFLINE = False
CLASSIFY_NEGITAVE = -14  # must < 0
CLASSIFY_POSTIVE = 99  # 99， 75，53， 36
classify_type = CLASSIFY_POSTIVE - CLASSIFY_NEGITAVE + 1

if __name__ == '__main__':
    init_log()
    logging.info("----------new grid search--------")
    # sys.stdout = open('start.log', 'w')

    start = datetime.datetime.now()
    logging.info("start at: %s" % start.strftime('%Y-%m-%d %H:%M:%S'))

    path = '/Users/a_piao/PycharmProjects/my_competition/NFLBigDataBowl/cache_feature.csv'

    if TRAIN_OFFLINE:
        if os.path.exists(path):
            train_basetable = pd.read_csv(path)
        else:
            train = pd.read_csv('../data/train.csv', dtype={'WindSpeed': 'object'})
            outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()
            train_basetable = create_features(train, False)
            train_basetable = process_two(train_basetable)
            train_basetable.to_csv(path, index=False)
    else:
        # train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
        train = pd.read_csv('/home/aistudio/data/data16525/train.csv', dtype={'WindSpeed': 'object'}) # 163
        # train = pd.read_csv('/home/aistudio/data/data16375/train.csv', dtype={'WindSpeed': 'object'}) #phone
        logging.info(train.shape)
        outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()
        train_basetable = create_features(train, False)
        train_basetable = process_two(train_basetable)

    X = train_basetable.copy()
    yards = X.Yards
    y = np.zeros((yards.shape[0], classify_type))
    for idx, target in enumerate(list(yards)):
        y[idx][-CLASSIFY_NEGITAVE + target] = 1
    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    params = {
        # 'n_estimators': 500,
        'learning_rate': 0.1,

        'num_leaves': 12,  # Original 50
        'max_depth': 4,

        'min_data_in_leaf': 91,  # min_child_samples
        'max_bin': 58,
        'min_child_weight': 7,

        "feature_fraction": 0.8,  # 0.9 colsample_bytree
        "bagging_freq": 1,
        "bagging_fraction": 0.8,  # 'subsample'
        "bagging_seed": 2019,

        'min_split_gain': 0.0,
        "lambda_l1": 0.8,
        "lambda_l2": 0.6,

        "boosting": "gbdt",
        'num_class': classify_type,  # 199 possible places
        # 'num_class': 199,  # 199 possible places
        'objective': 'multiclass',
        "metric": "None",
        # "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 2019,
    }
    find_best_param(X, y, params)

    # logging.info("调参1：提高准确率")
    # for num_leaves in range(5, 100, 5):
    #     for max_depth in range(3, 8, 1):
    #         params['num_leaves'] = num_leaves
    #         params['max_depth'] = max_depth
    #         find_best_param(X, y, params)
    # params = best_param

    # logging.info("调参1：提高准确率")
    # for max_depth in range(4, 10, 1):
    #     limit = min(pow(2, max_depth) + 1,100)
    #     for num_leaves in range(2, limit, 5):
    #         params['num_leaves'] = num_leaves
    #         params['max_depth'] = max_depth
    #         find_best_param(X, y, params)
    #     logging.info("current max depth: %d" % max_depth)
    # params = best_param

    logging.info("调参2：降低过拟合")
    for max_bin in range(28, 128, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            find_best_param(X, y, params)
        logging.info("current max_bin: %d" % max_bin)
    params = best_param

    # logging.info("调参2.1：降低过拟合")
    # for min_child_weight in range(1, 50, 5):
    #         params['min_child_weight'] = min_child_weight
    #         find_best_param(X, y, params)
    # params = best_param

    # logging.info("调参3：降低过拟合")
    # for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    #     for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    #         for bagging_freq in range(0, 50, 5):
    #             params['feature_fraction'] = feature_fraction
    #             params['bagging_fraction'] = bagging_fraction
    #             params['bagging_freq'] = bagging_freq
    #             find_best_param(X, y, params)
    #     logging.info("current feature_fraction: %f" % feature_fraction)
    # params = best_param
    #
    # logging.info("调参4：降低过拟合")
    # for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    #     for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
    #         params['lambda_l1'] = lambda_l1
    #         params['lambda_l2'] = lambda_l2
    #         find_best_param(X, y, params)
    # params = best_param
    #
    # logging.info("调参5：降低过拟合2")
    # for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     params['min_split_gain'] = min_split_gain
    #     find_best_param(X, y, params)
    # params = best_param

    logging.info("final best param: %s" % best_param)
    logging.info("final best score: %f" % best_score)

    end = datetime.datetime.now()
    logging.info("end at: %s" % end.strftime('%Y-%m-%d %H:%M:%S'))
    logging.info("during:%s\n" % str((end - start)).split('.')[0])
