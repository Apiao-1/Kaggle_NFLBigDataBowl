# https://www.kaggle.com/enzoamp/nfl-lightgbm/code
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

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def early_metric_crps(y_true, y_pred):
    print(y_pred.shape, y_true.shape)
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


def lgbm_grid_search():
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.1,

        'num_leaves': 190,  # Original 50
        'max_depth': 9,

        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 30,
        'subsample': .8,
        'colsample_bytree': .8,
        # "boosting": "gbdt",
        "feature_fraction": 0.8,  # 0.9
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "bagging_seed": 42,
        "lambda_l1": 0.8,
        "lambda_l2": 0.6,

        'num_class': 199,  # 199 possible places
        'objective': 'multiclass',
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42,
    }

    steps = {
        # 'n_estimators': {'step': 50, 'min': 1, 'max': 'inf'},
        'num_leaves': {'step': 10, 'min': 10, 'max': 200},
        'max_depth': {'step': 1, 'min': 3, 'max': 10},

        'max_bin': {'step': 5, 'min': 5, 'max': '250'},
        'min_data_in_leaf': {'step': 10, 'min': 1, 'max': 100},
        'feature_fraction': {'step': 0.1, 'min': 0.6, 'max': 1},
        'bagging_fraction': {'step': 0.1, 'min': 0.6, 'max': 1},
        # 'bagging_freq': {'step': 10, 'min': 1, 'max': 80},
        'bagging_freq': {'step': 10, 'min': 1, 'max': 80},
        'lambda_l1': {'step': 0.01, 'min': 0, 'max': 0.1},
        'lambda_l2': {'step': 0.01, 'min': 0, 'max': 0.1},
    }

    grid_search_auto(steps, params)


def grid_search_auto(steps, params):
    old_params = params.copy()

    print(params)

    while 1:
        print('---------------new grid search for all params------------------')
        # for循环调整各个参数达到局部最优
        for name, step in steps.items():
            score = -99999

            start = params[name] - step['step']
            if start <= step['min']:
                start = step['min']

            stop = params[name] + step['step']
            if step['max'] != 'inf' and stop >= step['max']:
                stop = step['max']
            # 调整单个参数达到局部最优
            while 1:

                if str(step['step']).count('.') == 1:
                    stop += step['step'] / 10
                else:
                    stop += step['step']

                param_grid = {
                    # 最开始这里会产生3个搜索的参数，begin +- step，但之后的轮次只会根据方向每次搜索一个参数
                    name: np.arange(start, stop, step['step']),
                }
                best_params, best_score = grid_search(param_grid)

                # 找到最优参数或者当前的得分best_score已经低于上一轮的score时结束(这里的低于或高于看具体的评价函数)
                if best_params[name] == params[name]:
                    print("got best params, round over:", params)
                    break

                if score > best_score:
                    print("score > best_score, round over:", params)
                    break

                # 当产生比当前结果更优的解时，则在此方向是继续寻找
                direction = (best_params[name] - params[name]) // abs(best_params[name] - params[name])  # 决定了每次只能跑两个参数
                start = stop = best_params[name] + step['step'] * direction  # 根据方向让下一轮每次搜索一个参数

                params[name] = best_params[name]

                if best_params[name] - step['step'] < step['min'] or (
                        step['max'] != 'inf' and best_params[name] + step['step'] > step['max']):
                    print("reach params limit, round over.", params)
                    break

                if abs(best_score - score) < abs(.0001 * score):
                    print("abs(best_score - score) < abs(.0001 * score), round over:", best_score - score, params)
                    score = best_score
                    print("update best score, best score = :", score)
                    break

                score = best_score
                print("update best score, best score = :", score)

                print("Next round search: ", params)

        # 最外层的while控制全局最优，任何一个参数发生了变化就要重新再调整，达到全局最优
        if old_params == params:
            break
        old_params = params

    print('grid search: %r\n' % (params))


TRAIN_OFFLINE = True


def grid_search(param):
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

    models = []
    kf = KFold(n_splits=5, random_state=42)
    score = []

    for i, (tdx, vdx) in enumerate(kf.split(X, y)):
        print(f'Fold : {i}')
        X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
        y_true = y_val.copy()

        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)
        # print(X_train.shape, y_train.shape) # (800, 199)
        # print(X_val.shape, y_val.shape) # (800, 199)

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        num_round = 1000
        model = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                          early_stopping_rounds=40)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        # print(y_pred.shape) # (200, 199)
        score_ = metric_crps(y_true, y_pred)
        # print(score_)
        score.append(score_)
    mean_score = np.mean(score)
    return -mean_score

if __name__ == '__main__':
    lgbm_grid_search()