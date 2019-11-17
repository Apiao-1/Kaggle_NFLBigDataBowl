# https://www.kaggle.com/enzoamp/nfl-lightgbm/code
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import datetime
import warnings
import os
import re

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def early_metric_crps(model, X_test, y_test):
    y_true = np.clip(np.cumsum(y_test, axis=1), 0, 1)
    y_pred = model.predict(X_test)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    crps = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])
    # print(crps)
    return -crps  # 要求是浮点数，并且分数越高越好（因此加了负号）


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


def grid_search_rf(criterion='gini', get_param=False):
    params = {
        'n_estimators': 450,
        'min_samples_split': 5,
        'min_samples_leaf': 13,
        'max_features': 0.4,
        'max_depth': 10,

        'bootstrap': False,
        # 'verbose':1,
        # 'criterion': 'gini',
        'random_state': 42
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 50, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
        'max_features': {'step': 0.1, 'min': 0.1, 'max': 1},
    }

    grid_search_auto(steps, params, RandomForestRegressor())


def grid_search_auto(steps, params, estimator):
    global log

    old_params = params.copy()

    print(params)

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

            best_params, best_score = grid_search(estimator.set_params(**params), param_grid)

            # 找到最优参数或者当前的得分best_score已经低于上一轮的score时结束(这里的低于或高于看具体的评价函数)
            if best_params[name] == params[name]:
                print("got best params, round over:", estimator.__class__.__name__, params)
                break

            if score > best_score:
                print("score > best_score, round over:", estimator.__class__.__name__, params)
                break

            # 当产生比当前结果更优的解时，则在此方向是继续寻找
            direction = (best_params[name] - params[name]) // abs(best_params[name] - params[name])  # 决定了每次只能跑两个参数
            start = stop = best_params[name] + step['step'] * direction  # 根据方向让下一轮每次搜索一个参数

            params[name] = best_params[name]

            if best_params[name] - step['step'] < step['min'] or (
                    step['max'] != 'inf' and best_params[name] + step['step'] > step['max']):
                print("reach params limit, round over.", estimator.__class__.__name__, params)
                break

            if abs(best_score - score) < abs(.0001 * score):
                print("abs(best_score - score) < abs(.0001 * score), round over:", best_score - score,
                      estimator.__class__.__name__, params)
                score = best_score
                print("update best score, best score = :", score)
                break

            score = best_score
            print("update best score, best score = :", score)

            print("Next round search: ", estimator.__class__.__name__, params)

    old_params = params

    print('grid search: %s\n%r\n' % (estimator.__class__.__name__, params))

def grid_search(estimator, param_grid):
    start = datetime.datetime.now()

    print('-----------search single param begin-----------')
    # print(start.strftime('%Y-%m-%d %H:%M:%S'))
    print(param_grid)
    print()

    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
    global outcomes
    outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()
    train_basetable = create_features(train, False)

    X = train_basetable
    yards = X.Yards

    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1
    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # data = feature.get_train_tree_data()

    estimator_name = estimator.__class__.__name__
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    if estimator_name is not 'XGBClassifier' and estimator_name is not 'LGBMClassifier':
        X_train = X
        y_train = y

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # clf = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs, cv=5)
    # clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=n_jobs,cv=5)
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=early_metric_crps, cv=5)

    clf = fit_eval_metric(clf, X_train, y_train, estimator_name, X_test=X_val, y_test=y_val)  # 拟合评估指标

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.10f (+/-%0.05f) for %r' % (mean, std * 2, params))
    print()
    print('best params', clf.best_params_)
    print('best score', clf.best_score_)
    print('time: %s' % str((datetime.datetime.now() - start)).split('.')[0])
    print('-----------search single param end-----------')
    print()

    return clf.best_params_, clf.best_score_


def fit_eval_metric(estimator, X, y, name=None, X_test=None, y_test=None):
    if name is None:
        name = estimator.__class__.__name__

    if X_test is None:
        estimator.fit(X, y)
    else:
        if name is 'XGBClassifier' or name is 'LGBMClassifier':
            print("early stopping")
            estimator.fit(X, y, eval_set=[(X, y), (X_test, y_test)], early_stopping_rounds=40,
                          eval_metric=early_metric_crps)
        else:
            estimator.fit(X, y)

    return estimator


if __name__ == '__main__':
    grid_search_rf()
