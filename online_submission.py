import matplotlib as mpl
import seaborn as sns
# from kaggle.competitions import nflrush
import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
import keras

from tqdm import tqdm_notebook
import warnings
import datetime
import os
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier

log = ''
cpu_jobs = -1


def init_setting():
    warnings.filterwarnings('ignore')
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    sns.set_style('darkgrid')
    mpl.rcParams['figure.figsize'] = [15, 10]
    global log


#     log = logger.init_logger()


# 提交的作品将根据连续排列的概率分数(CRPS)进行评估。
# 对于每个PlayId，您必须预测获得或丢失码数的累积概率分布。换句话说，您预测的每一列表示该队在比赛中获得<=那么多码的概率。
def return_step(x):
    temp = np.zeros(199)
    temp[x + 99:] = 1
    return temp


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


def preprocess(train, online=False):
    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    train["GameClock_minute"] = train["GameClock"].apply(lambda x: x.split(":")[0]).astype("object")  # hour

    ## Height
    train['PlayerHeight_dense'] = train['PlayerHeight'].apply(
        lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    ## Age
    seconds_in_year = 60 * 60 * 24 * 365.25
    train['PlayerAge'] = train.apply(
        lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

    ## WindSpeed
    # print(train['WindSpeed'].value_counts())
    train['WindSpeed_ob'] = train['WindSpeed'].apply(
        lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(
        lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(
        lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(x) and type(
            x) != float and 'gusts up to' in x else x)
    train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)

    ## Weather
    train['GameWeather_process'] = train['GameWeather'].str.lower()
    train['GameWeather_process'] = train['GameWeather_process'].apply(
        lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(
        lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(
            x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(
        lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather_process'] = train['GameWeather_process'].apply(
        lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)

    ## Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team": "RusherTeam"})
    train = train.merge(temp, on="PlayId")
    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    ## dense -> categorical
    train["Quarter_ob"] = train["Quarter"].astype("object")
    train["Down_ob"] = train["Down"].astype("object")
    train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")
    train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    # train["Week_ob"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")

    ## Orientation and Dir
    train["Orientation_ob"] = train["Orientation"].apply(lambda x: orientation_to_cat(x)).astype("object")
    train["Dir_ob"] = train["Dir"].apply(lambda x: orientation_to_cat(x)).astype("object")

    train["Orientation_sin"] = train["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    train["Orientation_cos"] = train["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
    train["Dir_sin"] = train["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    train["Dir_cos"] = train["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype(
        "object")

    ## Turf
    Turf = {'Field Turf': 'Artificial', 'A-Turf Titan': 'Artificial', 'Grass': 'Natural',
            'UBU Sports Speed S5-M': 'Artificial', 'Artificial': 'Artificial', 'DD GrassMaster': 'Artificial',
            'Natural Grass': 'Natural', 'UBU Speed Series-S5-M': 'Artificial', 'FieldTurf': 'Artificial',
            'FieldTurf 360': 'Artificial', 'Natural grass': 'Natural', 'grass': 'Natural', 'Natural': 'Natural',
            'Artifical': 'Artificial', 'FieldTurf360': 'Artificial', 'Naturall Grass': 'Natural',
            'Field turf': 'Artificial', 'SISGrass': 'Artificial', 'Twenty-Four/Seven Turf': 'Artificial',
            'natural grass': 'Natural'}
    train['Turf'] = train['Turf'].map(Turf)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(
        lambda x: pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on="PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(
        lambda x: pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on="PlayId")

    if (online == False):
        train = filter_by_times(train)

    ## sort
    #     train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by=['X']).sort_values(by=['Dis']).sort_values(
        by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop=True)
    # pd.to_pickle(train, "train.pkl")
    drop(train)
    return train


def filter_by_times(train):
    ## DisplayName remove Outlier
    v = train["DisplayName"].value_counts()
    missing_values = list(v[v < 5].index)
    train["DisplayName"] = train["DisplayName"].where(~train["DisplayName"].isin(missing_values),
                                                      "nan")  # 如果 cond 为真，保持原来的值，否则替换为other

    ## PlayerCollegeName remove Outlier
    v = train["PlayerCollegeName"].value_counts()
    missing_values = list(v[v < 10].index)
    train["PlayerCollegeName"] = train["PlayerCollegeName"].where(~train["PlayerCollegeName"].isin(missing_values),
                                                                  "nan")
    return train


cat_features = []  # 标签型的
categories = []
most_appear_each_categories = {}
dense_features = []  # 数值型的
le = LabelEncoder()
sss = {}
medians = {}


def split_dense_cat_feature(train, online=False):
    if (online == False):
        for col in train.columns:
            if train[col].dtype == 'object':
                cat_features.append(col)
                # print("*cat*", col, len(train[col].unique()))
            else:
                dense_features.append(col)
                # print("!dense!", col, len(train[col].unique()))
        dense_features.remove("Yards")

    dense_features.remove("PlayId")

    # categorical
    global categories
    train_cat = train[cat_features]
    for col in tqdm_notebook(train_cat.columns):
        train_cat.loc[:, col] = train_cat[col].fillna("nan")
        train_cat.loc[:, col] = col + "__" + train_cat[col].astype(str)
        most_appear_each_categories[col] = list(train_cat[col].value_counts().index)[0]  # 取类别最多的
        categories.append(train_cat[col].unique())
    categories = np.hstack(categories)  # 所有不同种类的类别
    print(len(categories))

    le.fit(categories)
    # Label Encode,转化为数字标签
    for col in tqdm_notebook(train_cat.columns):
        train_cat.loc[:, col] = le.transform(train_cat[col])
    num_classes = len(le.classes_)
    print(num_classes)

    # Dense
    train_dense = train[dense_features]
    for col in tqdm_notebook(train_dense.columns):
        medians[col] = np.nanmedian(train_dense[col])  # 忽略Nan值后的中位数
        train_dense.loc[:, col] = train_dense[col].fillna(medians[col])
        ss = StandardScaler()
        train_dense.loc[:, col] = ss.fit_transform(train_dense[col].values[:, None])
        sss[col] = ss
    return train_dense, train_cat, num_classes, cat_features, dense_features


def drop(train):
    drop_cols = ["GameId", "GameWeather", "NflId", "Season", "NflIdRusher"]
    drop_cols += ['TimeHandoff', 'TimeSnap', 'PlayerBirthDate']
    drop_cols += ["Orientation", "Dir", 'WindSpeed', "GameClock"]
    drop_cols += ["DefensePersonnel", "OffensePersonnel"]
    train.drop(drop_cols, axis=1, inplace=True)
    return train


dense_player_features = dense_game_features = cat_game_features = cat_player_features = []


def divide_dense_cat_columns(train_dense, train_cat):
    # Divide features into groups
    global dense_player_features, dense_game_features, cat_game_features, cat_player_features
    ## dense features for play, 同一队伍里std为0即作为整个play的特征 (5,) ，如果是11则(11,)
    dense_game_features = train_dense.columns[train_dense[:22].std() == 0]
    ## dense features for each player (47,) 如果是11则(41,)
    dense_player_features = train_dense.columns[train_dense[:22].std() != 0]
    ## categorical features for play (14,) 如果是11则(15,)
    cat_game_features = train_cat.columns[train_cat[:22].std() == 0]
    ## categorical features for each player (5,) 如果是11则(4,)
    cat_player_features = train_cat.columns[train_cat[:22].std() != 0]


def get_NN_feature(train_dense, train_cat):
    # 23170*5 ,23170*22 = 总数据量，这里已经做了压缩
    train_dense_game = train_dense[dense_game_features].iloc[np.arange(0, len(train_dense), 22)].reset_index(drop=True)
    # train_dense_game = train_dense[dense_game_features].iloc[np.arange(0, len(train_dense), 22)].reset_index(drop=True).values
    ## with rusher player feature，22个人中只有一个是rusher，因此可以把rusher特征当成整个play的特征
    train_dense_game = pd.concat(
        [train_dense_game, train_dense[dense_player_features][train_dense["IsRusher"] > 0].reset_index(drop=True)],
        axis=1)
    # train_dense_game = np.hstack([train_dense_game, train_dense[dense_player_features][train_dense["IsRusher"] > 0]])

    train_dense_players = [
        train_dense[dense_player_features].iloc[np.arange(k, len(train_dense), 22)].reset_index(drop=True)
        for k in range(22)]
    train_dense_players = np.stack([t.values for t in train_dense_players]).transpose(1, 0,
                                                                                      2)  # 通过transpose()函数改变了x的索引值为（1，0，2），对应（y，x，z）

    train_cat_game = train_cat[cat_game_features].iloc[np.arange(0, len(train_dense), 22)].reset_index(drop=True)
    # train_cat_game = train_cat[cat_game_features].iloc[np.arange(0, len(train_dense), 22)].reset_index(drop=True).values
    train_cat_game = pd.concat(
        [train_cat_game, train_cat[cat_player_features][train_dense["IsRusher"] > 0].reset_index(drop=True)],
        axis=1)  ## with rusher player feature
    # [train_cat_game, train_cat[cat_player_features][train_dense["IsRusher"] > 0]])  ## with rusher player feature

    train_cat_players = [train_cat[cat_player_features].iloc[np.arange(k, len(train_dense), 22)].reset_index(drop=True)
                         for k in range(22)]
    train_cat_players = np.stack([t.values for t in train_cat_players]).transpose(1, 0, 2)
    return train_dense_game, train_dense_players, train_cat_game, train_cat_players


def get_train_label(train):
    train_y = train["Yards"].iloc[np.arange(0, len(train), 22)].reset_index(drop=True)
    train_y_199 = np.vstack(train_y.apply(return_step).values)
    return train_y, train_y_199


def get_train_NN_data():
    path = 'cache_feature_train.csv'  # cache_NN

    if os.path.exists(path):
        train = pd.read_csv(path)
        # print(len(data))
    else:
        train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
        # train = pd.read_csv('data/train.csv', dtype={'WindSpeed': 'object'})
        train = preprocess(train)
        train.to_csv(path, index=False)

    train_dense, train_cat, num_classes_cat, _, _ = split_dense_cat_feature(train)
    print(train_dense.shape, train_cat.shape, train.shape)  # (509762, 52) (509762, 19) (509762, 73)
    divide_dense_cat_columns(train_dense, train_cat)
    train_dense_game, train_dense_players, train_cat_game, train_cat_players = get_NN_feature(train_dense, train_cat)
    # print(train_dense_game.shape, train_dense_players.shape, train_cat_game.shape, train_cat_players.shape) # (23171, 52) (23171, 22, 47) (23171, 19) (23171, 22, 5)

    train_y, train_y_199 = get_train_label(train)
    # print(train_y_raw.shape, train_y.shape) # (23171,) (23171, 199)

    return train_dense_game, train_dense_players, train_cat_game, train_cat_players, num_classes_cat, train_y, train_y_199


def get_train_tree_data():
    path = 'cache_tree_train.csv'

    if os.path.exists(path):
        train_x_y = pd.read_csv(path)
        print("load train tree data from disk with shape: ", train_x_y.shape)
    else:
        train_dense_game, _, train_cat_game, _, _, train_y, train_y_199 = get_train_NN_data()
        train_x_y = pd.concat([train_dense_game, train_cat_game, train_y], axis=1)  # (23171, 71)
        # train_x = pd.concat([train_dense_game, train_cat_game], axis=1, ignore_index=True) # (23171, 71)
        # train_x = pd.merge(train_dense_game, train_cat_game, how='left', on='PlayId')
        train_x_y.to_csv(path, index=False)

    return train_x_y


def fit_eval_metric(estimator, X, y, name=None, X_test=None, y_test=None):
    if name is None:
        name = estimator.__class__.__name__

    # if name is 'XGBClassifier' or name is 'LGBMRegressor':
    #     estimator.fit(X, y, eval_metric='mae')
    # else:
    #     estimator.fit(X, y)

    if X_test is None:
        estimator.fit(X, y)
    else:
        # train_data, test_data = train_test_split(X,shuffle=True)
        # X = train_data
        # y = train_data.pop('Yards')
        # X_test = test_data
        # y_test = X_test.pop('Yards')
        if name is 'XGBClassifier' or name is 'LGBMRegressor':
            estimator.fit(X, y, eval_set=[(X, y), (X_test, y_test)], early_stopping_rounds=100)
        else:
            estimator.fit(X, y)
    # results = estimator.evals_result
    # epochs = len(results['validation_0']['mae'])
    # x_axis = range(0, epochs)
    # plot log loss
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    # ax.plot(x_axis, results['validation_1']['mae'], label='Test')
    # ax.legend()
    # pyplot.ylabel('mae')
    # pyplot.title('lgbm mae')
    # pyplot.show()

    return estimator


def grid_search(estimator, param_grid):
    start = datetime.datetime.now()

    print('-----------search single param begin-----------')
    # print(start.strftime('%Y-%m-%d %H:%M:%S'))
    print(param_grid)
    print()

    data = get_train_tree_data()

    data, _ = train_test_split(data, random_state=0)

    # X = data.copy().drop(columns='Coupon_id')
    X = data.copy()
    y = X.pop('Yards')

    estimator_name = estimator.__class__.__name__
    n_jobs = cpu_jobs
    # if estimator_name is 'XGBClassifier' or estimator_name is 'LGBMClassifier' or estimator_name is 'CatBoostClassifier':
    #     n_jobs = 1

    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=n_jobs,
                       cv=5
                       )

    clf = fit_eval_metric(clf, X, y, estimator_name)  # 拟合评估指标

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.5f (+/-%0.05f) for %r' % (mean, std * 2, params))
    print()
    print('best params', clf.best_params_)
    print('best score', clf.best_score_)
    print('time: %s' % str((datetime.datetime.now() - start)).split('.')[0])
    print('-----------search single param end-----------')
    print()

    return clf.best_params_, clf.best_score_


# Pipeline 自动化 Grid Search，只要预先设定好使用的 Model 和参数的候选，就能自动搜索并记录最佳的 Model。
def grid_search_auto(steps, params, estimator):
    global log

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

                best_params, best_score = grid_search(estimator.set_params(**params), param_grid)

                # 找到最优参数或者当前的得分best_score已经低于上一轮的score时结束
                if best_params[name] == params[name] or score > best_score:
                    print("got best params, round over:", estimator.__class__.__name__, params)
                    break

                # 当产生比当前结果更优的解时，则在此方向是继续寻找
                direction = (best_params[name] - params[name]) // abs(best_params[name] - params[name])  # 决定了每次只能跑两个参数
                start = stop = best_params[name] + step['step'] * direction  # 根据方向让下一轮每次搜索一个参数

                score = best_score
                params[name] = best_params[name]

                if best_params[name] - step['step'] < step['min'] or (
                        step['max'] != 'inf' and best_params[name] + step['step'] > step['max']):
                    print("reach params limit, round over.", estimator.__class__.__name__, params)
                    break

                print("Next round search: ", estimator.__class__.__name__, params)

        # 最外层的while控制全局最优，任何一个参数发生了变化就要重新再调整，达到全局最优
        if old_params == params:
            break
        old_params = params

    print('grid search: %s\n%r\n' % (estimator.__class__.__name__, params))
    log += 'grid search: %s\n%r\n' % (estimator.__class__.__name__, params)


#     logger.set_logger(log)


def grid_search_gbdt(get_param=False):
    params = {
        # 10
        # 'learning_rate': 1e-2,
        # 'n_estimators': 1900,
        # 'max_depth': 9,
        # 'min_samples_split': 200,
        # 'min_samples_leaf': 50,
        # 'subsample': .8,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 150,
        # 'max_depth': 8,
        # 'min_samples_split': 200,
        # 'min_samples_leaf': 50,
        # 'subsample': .8,
        'learning_rate': 0.0125,
        'n_estimators': 700,
        'max_depth': 3,
        'min_samples_split': 190,
        'min_samples_leaf': 70,
        'subsample': .8,

        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 50, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        # 'min_samples_split': {'step': 10, 'min': 2, 'max': 'inf'},
        # 'min_samples_leaf': {'step': 10, 'min': 1, 'max': 'inf'},
        # 'subsample': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, GradientBoostingRegressor())


def grid_search_xgb(get_param=False):
    global cpu_jobs
    params = {
        # all
        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'num_boost_round': 3500,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,
        # 'scale_pos_weight': 1,
        # 'reg_alpha': 0,

        'n_jobs': cpu_jobs,
        'seed': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_child_weight': {'step': 1, 'min': 1, 'max': 'inf'},
        'gamma': {'step': .1, 'min': 0, 'max': 1},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
        'scale_pos_weight': {'step': 1, 'min': 1, 'max': 10},
        'reg_alpha': {'step': .1, 'min': 0, 'max': 1},
    }

    grid_search_auto(steps, params, XGBClassifier())


def grid_search_lgb(get_param=False):
    params = {
        # 10
        # 'learning_rate': 1e-2,
        # 'n_estimators': 1200,
        # 'num_leaves': 51,
        # 'min_split_gain': 0,
        # 'min_child_weight': 1e-3,
        # 'min_child_samples': 22,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        'learning_rate': .025,
        'n_estimators': 360,
        'num_leaves': 50,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 21,
        'subsample': .8,
        'colsample_bytree': .8,
        'n_jobs': cpu_jobs,
        'random_state': 0,
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'num_leaves': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_split_gain': {'step': .1, 'min': 0, 'max': 1},
        'min_child_weight': {'step': 1e-3, 'min': 1e-3, 'max': 'inf'},
        'min_child_samples': {'step': 1, 'min': 1, 'max': 'inf'},
        # 'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, LGBMRegressor())


def grid_search_cat(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 3600,
        'max_depth': 8,
        'max_bin': 127,
        'reg_lambda': 2,
        'subsample': .7,

        # 'learning_rate': 1e-1,
        # 'iterations': 460,
        # 'depth': 8,
        # 'l2_leaf_reg': 8,
        # 'border_count': 37,

        # 'ctr_border_count': 16,
        'one_hot_max_size': 2,
        'bootstrap_type': 'Bernoulli',
        'leaf_estimation_method': 'Newton',
        'random_state': 0,
        'verbose': False,
        'eval_metric': 'AUC',
        'thread_count': cpu_jobs
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'max_bin': {'step': 1, 'min': 1, 'max': 255},
        'reg_lambda': {'step': 1, 'min': 0, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'one_hot_max_size': {'step': 1, 'min': 0, 'max': 255},
    }

    grid_search_auto(steps, params, CatBoostClassifier())


def grid_search_rf(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3090,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0
        }
    else:
        params = {
            'n_estimators': 3110,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, RandomForestClassifier())


def grid_search_et(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3060,
            'max_depth': 22,
            'min_samples_split': 12,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0,
        }
    else:
        params = {
            'n_estimators': 3100,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, ExtraTreesClassifier())


def train_gbdt(model=False):
    global log

    params = grid_search_gbdt(True)
    clf = GradientBoostingClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'gbdt'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', subsample: %.1f' % params['subsample']
    log += '\n\n'

    return train(clf)


def train_xgb(model=False):
    global log

    params = grid_search_xgb(get_param=True)

    clf = XGBClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'xgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_child_weight: %d' % params['min_child_weight']
    log += ', gamma: %.1f' % params['gamma']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_lgb(model=False):
    global log

    params = grid_search_lgb(True)

    clf = LGBMRegressor().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'lgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', num_leaves: %d' % params['num_leaves']
    log += ', min_split_gain: %.1f' % params['min_split_gain']
    log += ', min_child_weight: %.4f' % params['min_child_weight']
    log += ', min_child_samples: %d' % params['min_child_samples']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_cat(model=False):
    global log

    params = grid_search_cat(True)

    clf = CatBoostClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'cat'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', iterations: %d' % params['iterations']
    log += ', depth: %d' % params['depth']
    log += ', l2_leaf_reg: %d' % params['l2_leaf_reg']
    log += ', border_count: %d' % params['border_count']
    log += ', subsample: %d' % params['subsample']
    log += ', one_hot_max_size: %d' % params['one_hot_max_size']
    log += '\n\n'

    return train(clf)


def train_rf(clf):
    global log

    params = clf.get_params()
    log += 'rf'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_rf_gini(model=False):
    clf = RandomForestClassifier().set_params(**grid_search_rf('gini', True))
    if model:
        return clf
    return train_rf(clf)


def train_rf_entropy():
    clf = RandomForestClassifier().set_params(**grid_search_rf('entropy', True))

    return train_rf(clf)


def train_et(clf):
    global log

    params = clf.get_params()
    log += 'et'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_et_gini(model=False):
    clf = ExtraTreesClassifier().set_params(**grid_search_et('gini', True))
    if model:
        return clf
    return train_et(clf)


def train_et_entropy():
    clf = ExtraTreesClassifier().set_params(**{
        'n_estimators': 3100,
        'max_depth': 13,
        'min_samples_split': 70,
        'min_samples_leaf': 10,
        'criterion': 'entropy',
        'random_state': 0
    })

    return train_et(clf)


def CRPS(y_valid_pred, y_train=None):
    y_pred = np.zeros((y_valid_pred.shape[0], 199))

    for i, p in enumerate(np.round(y_valid_pred)):
        p += 99
        for j in range(199):
            if j >= p + 10:
                y_pred[i][j] = 1.0
            elif j >= p - 10:
                y_pred[i][j] = (j + 10 - p) * 0.05
    print(y_pred.shape) # （1，199）
    if y_train is None:
        return y_pred

    global log
    y_true = np.zeros((y_train.shape[0], 199))
    if y_true.shape != y_pred.shape:
        print("ERROR, y_true.shape != y_pred.shape:")
        exit()

    for i, p in enumerate(y_train):
        p += 99
        for j in range(199):
            if j >= p:
                y_true[i][j] = 1.0

    crps = np.sum(np.power(y_pred - y_true, 2)) / (199 * (509762 // 22))
    print("CRPS:", crps)
    log += '  CRPS: %f\n' % crps
    return y_pred


# train the model, clf = classifier分类器
def train(clf):
    global log

    data = get_train_tree_data()

    train_data, test_data = train_test_split(data,
                                             # train_size=1000,
                                             # train_size=100000,
                                             random_state=0,
                                             shuffle=True
                                             )

    # _, test_data = train_test_split(data, random_state=0)

    X_train = train_data.copy()
    y_train = X_train.pop('Yards')

    X_test = test_data.copy()
    y_test = X_test.pop('Yards')

    clf = fit_eval_metric(clf, X_train, y_train, X_test, y_test)

    y_true, y_pred = y_test, clf.predict(X_test)
    # log += '%s\n' % classification_report(y_test, y_pred)
    log += '  mean_squared_error: %f\n' % mean_squared_error(y_true, y_pred)
    # y_score = clf.predict_proba(X_test)[:, 1]  # 这里可以尝试用累加再截断，待测试
    # print(mean_squared_error(y_true, y_pred))
    # print(mean_absolute_error(y_true, y_pred))
    log += '  mean_absolute_error: %f\n' % mean_absolute_error(y_true, y_pred)

    # y_pred = CRPS(y_true, y_pred)
    y_pred = CRPS(y_pred, y_true)
    print("y_pred.shape:", y_pred.shape)

    #     logger.set_logger(log)

    return clf



def local_cv_eval(model, k=5):
    data = get_train_tree_data()
    y = data.pop('Yards')
    clf = eval('train_%s' % model)()

    # losses = []
    # models = []
    for i in range(1):
        scores_cv = cross_val_score(clf, data, y, scoring='mean_absolute_error', cv=k)
        print("MAE: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))


def online_submit(data, model):
    test = preprocess(data, online=True)

    ### categorical
    test_cat = test[cat_features]
    for col in (test_cat.columns):
        test_cat.loc[:, col] = test_cat[col].fillna("nan")
        test_cat.loc[:, col] = col + "__" + test_cat[col].astype(str)
        isnan = ~test_cat.loc[:, col].isin(categories)
        if np.sum(isnan) > 0:
            if not ((col + "__nan") in categories):
                test_cat.loc[isnan, col] = most_appear_each_categories[col]
            else:
                test_cat.loc[isnan, col] = col + "__nan"
    for col in (test_cat.columns):
        test_cat.loc[:, col] = le.transform(test_cat[col])

    ### dense
    test_dense = test[dense_features]
    for col in (test_dense.columns):
        test_dense.loc[:, col] = test_dense[col].fillna(medians[col])
        test_dense.loc[:, col] = sss[col].transform(test_dense[col].values[:, None])

    print(test_dense.shape, test_cat.shape, test.shape) # (22, 42) (22, 29) (22, 72)

    ### divide
    train_dense_game, _, train_cat_game, _ = get_NN_feature(test_dense, test_cat)

    print(train_dense_game.shape, train_cat_game.shape) # (1, 42) (1, 29)
    ## pred
    train_x_y = pd.concat([train_dense_game, train_cat_game], axis=1)
    y_pred = model.predict(train_x_y)
    y_pred = CRPS(y_pred)
    return y_pred


init_setting()
env = nflrush.make_env()
model = train_lgb()

for (test_df, sample_prediction_df) in tqdm.tqdm_notebook(env.iter_test()):
    pred = online_submit(test_df, model)
    env.predict(pd.DataFrame(data=pred, columns=sample_prediction_df.columns))
env.write_submission_file()
