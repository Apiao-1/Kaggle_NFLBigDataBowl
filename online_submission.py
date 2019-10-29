import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, KFold, GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import sqlite3
import xgboost as xgb
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization
import warnings
from string import punctuation
import re
import math

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 200)


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
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0, 'QB': 0, 'RB': 0, 'TE': 0, 'WR': 0}
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


def transform_time_all(str1, quarter):
    if quarter <= 4:
        return 15 * 60 - (int(str1[:2]) * 60 + int(str1[3:5])) + (quarter - 1) * 15 * 60
    if quarter == 5:
        return 10 * 60 - (int(str1[:2]) * 60 + int(str1[3:5])) + (quarter - 1) * 15 * 60


def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,
                              range=(-99, 100), density=True)
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T,
                          columns=['Yards' + str(i) for i in range(-99, 100)])
    return cdf_df


def get_score_pingyi1(y_pred, y_true, cdf, w, dist_to_end):
    y_pred = int(y_pred)
    if y_pred == w:
        y_pred_array = cdf.copy()
    elif y_pred - w > 0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred - w):] = cdf[:(-(y_pred - w))].copy()
    elif w - y_pred > 0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred - w)] = cdf[(w - y_pred):].copy()
    y_pred_array[-1] = 1
    y_pred_array[(dist_to_end + 99):] = 1
    y_true_array = np.zeros(199)
    y_true_array[(y_true + 99):] = 1
    return np.mean((y_pred_array - y_true_array) ** 2)


def CRPS_pingyi1(y_preds, y_trues, w, cdf, dist_to_ends):
    if len(y_preds) != len(y_trues):
        print('length does not match')
        return None
    n = len(y_preds)
    tmp = []
    for a, b, c in zip(y_preds, y_trues, dist_to_ends):
        tmp.append(get_score_pingyi1(a, b, cdf, w, c))
    return np.mean(tmp)


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
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


def get_score(y_pred, cdf, w, dist_to_end):
    y_pred = int(y_pred)
    if y_pred == w:
        y_pred_array = cdf.copy()
    elif y_pred - w > 0:
        y_pred_array = np.zeros(199)
        y_pred_array[(y_pred - w):] = cdf[:(-(y_pred - w))].copy()
    elif w - y_pred > 0:
        y_pred_array = np.ones(199)
        y_pred_array[:(y_pred - w)] = cdf[(w - y_pred):].copy()
    y_pred_array[-1] = 1
    y_pred_array[(dist_to_end + 99):] = 1
    return y_pred_array


def euclidean_distance(x1, y1, x2, y2):
    x_diff = (x1 - x2) ** 2
    y_diff = (y1 - y2) ** 2

    return np.sqrt(x_diff + y_diff)

def min_tackle_time(dist, v, a):
    return (np.sqrt(v*v + 2 * a *dist) - v) / a


def drop(train):
    # drop_cols += ["Orientation", "Dir"]

    play_drop = ["GameId", 'PlayId', "TimeHandoff", "TimeSnap", "GameClock", "DefensePersonnel", "OffensePersonnel",
                 'FieldPosition', 'PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                 'HomeScoreBeforePlay','VisitorScoreBeforePlay','TeamOnOffense','Stadium']
    player_drop = ['DisplayName', 'PlayerBirthDate', "IsRusher", "NflId", "NflIdRusher", "Dir",
                   'Dir_rad', 'Ori_rad',"PlayDirection",'Orientation','Rusher_X','Rusher_Y',
                   'dist_to_rusher','time_to_rusher']
    environment_drop = ["WindSpeed", "WindDirection", "Season", "GameWeather",'Location','GameWeather_process',
                        'Turf']
    drop_cols = player_drop + play_drop + environment_drop
    train.drop(drop_cols, axis=1, inplace=True)
    return train

def preprocess(train):
    # fix some encode https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    # ——————— play ———————
    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    # train["GameClock_minute"] = train["GameClock"].apply(lambda x: x.split(":")[0])  # hour
    train['time_end'] = train.apply(lambda x: transform_time_all(x.loc['GameClock'], x.loc['Quarter']), axis=1)

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    # train['date_game'] = train_single.GameId.map(lambda x: pd.to_datetime(str(x)[:8]))

    ## play是否发生在控球方所在的半场
    train['own_field'] = (train['FieldPosition'].fillna('') == train['PossessionTeam']).astype(int)
    ## 主队持球或是客队持球
    train['process_type'] = (train['PossessionTeam'] == train['HomeTeamAbbr']).astype(int)

    ## PlayDirection
    train['ToLeft'] = train.PlayDirection == "left"
    # train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')

    # 是否为防守方
    train['TeamOnOffense'] = "home"
    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    train['IsOnOffense'] = train.Team == train.TeamOnOffense  # Is player on offense?

    # 发球线离自家球门的实际码线距离
    train['YardLine'] = train.apply(
        lambda x: (x.loc['YardLine']) if x.loc['own_field'] == 1 else (100 - x.loc['YardLine']), axis=1)
    # train['dist_to_end_train'] = train.apply(lambda x: (100 - x.loc['YardLine']) if x.loc['own_field'] == 1 else x.loc['YardLine'], axis=1)
    # ? https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    # train['dist_to_end_train'] = train.apply(lambda row: row['dist_to_end_train'] if row['PlayDirection'] else 100 - row['dist_to_end_train'],axis=1)
    # train.drop(train.index[(train['dist_to_end_train'] < train['Yards']) | (train['dist_to_end_train'] - 100 > train['Yards'])],inplace=True)


    # 统一进攻方向 https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
    # https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python?scriptVersionId=22014032
    train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi / 180.0
    train['Ori_rad'] = np.mod(90 - train.Orientation, 360) * math.pi / 180.0
    # train['X_std'] = train.X
    train.loc[train.ToLeft, 'X'] = 120 - train.loc[train.ToLeft, 'X']
    # train['Y_std'] = train.Y
    train.loc[train.ToLeft, 'Y'] = 160 / 3 - train.loc[train.ToLeft, 'Y']
    train['Dir_std'] = train.Dir_rad
    train['Ori_std'] = train.Ori_rad
    train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2 * np.pi)
    train.loc[train.ToLeft, 'Ori_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Ori_rad'], 2 * np.pi)

    # 离发球线距离x
    train['dist_yardline'] = train['YardLine'] - train['X']/0.91

    # 方向是否与进攻方向相同
    train['is_Dir_back'] = train['Dir_rad'].apply(lambda x: 1 if (x > np.pi) else 0)
    train['is_Ori_back'] = train['Ori_std'].apply(lambda x: 1 if (x > np.pi) else 0)
    train['Dir_std'] = train['Dir_std'].apply(lambda x: np.mod(x, np.pi))
    train['Ori_std'] = train['Ori_std'].apply(lambda x: np.mod(x, np.pi))

    # 分方向的速度
    train["Dir_std_sin"] = train["Dir_std"].apply(lambda x: np.sin(x))
    train["Dir_std_cos"] = train["Dir_std"].apply(lambda x: np.cos(x))
    train['S_horizontal'] = train['S'] * train['Dir_std_cos']
    train['S_vertical'] = train['S'] * train['Dir_std_sin']

    ## Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    # train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")
    # temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team": "RusherTeam"})
    # train = train.merge(temp, on="PlayId")
    # train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    # 球员距rusher的距离
    tmp = train[train['IsRusher'] == True][['GameId', 'PlayId', 'X', 'Y']].copy().rename(columns={'X': 'Rusher_X',
                                                                                                      'Y': 'Rusher_Y'})
    train = pd.merge(train, tmp, on=['GameId', 'PlayId'], how='inner')
    train['dist_to_rusher'] = train[['X', 'Y', 'Rusher_X', 'Rusher_Y']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

    # 所有球员跑向rusher需要的时间,(假设rusher不动)
    train['time_to_rusher'] = train[['X', 'Y','Rusher_X', 'Rusher_Y', 'S', 'A',]].apply(
            lambda x: min_tackle_time(euclidean_distance(x[0], x[1], x[2], x[3]), x[4], x[5]), axis=1)
    # train['time_to_rusher_Defend'] = train[train['IsOnOffense'] == False][['X', 'Y','Rusher_X', 'Rusher_Y', 'S', 'A',]].apply(
    #         lambda x: min_tackle_time(euclidean_distance(x[0], x[1], x[2], x[3]), x[4], x[5]), axis=1)

    # Rusher距QB的距离
    QB_distance = train[train['Position'] == 'QB'][['dist_to_rusher','GameId','PlayId']].rename(columns={'dist_to_rusher':'dist_QB'})
    train =  pd.merge(train, QB_distance, on=['GameId', 'PlayId'], how='inner')

    # let's say now I want for that specific play to have as features the # of players within 3, 6, 9, 12, 15
    # yards of distance from the runner. In that case, as I already have the distances from the runner to each of the 11 defense players, I will count how many of them are within each of these intervals, and return those.
    # defense_x = train[train['IsOnOffense'] == False][['X','Rusher_X','GameId','PlayId']].apply

    # 每个play对应两条,敌方球员距离，友方球员距离
    Offense_player_distance = train[(train['IsOnOffense'] == True) & (train['dist_to_rusher'] > 0)].groupby(['GameId', 'PlayId'])\
        .agg({'dist_to_rusher': ['min', 'max', 'mean', 'std'],
              'X': ['mean', 'std'], 'Y': ['max','min','mean', 'std'],
              'time_to_rusher':['mean','min']
              }).rename(columns={
        'min': 'Offense_min','max':'Offense_max','mean':'Offense_mean', 'std':'Offense_std'}).reset_index()
    Defense_player_distance = train[train['IsOnOffense'] == False].groupby(['GameId', 'PlayId'])\
        .agg({'dist_to_rusher': ['min', 'max', 'mean', 'std'],
              'X':['mean','std'],'Y':['mean','std','max','min'],
              'time_to_rusher':['mean','min']
              }).reset_index() # min表示防守方跑的最快的球员跑到rusher的时间
    Defense_player_distance['defense_y_spread'] = Defense_player_distance[('Y','max')] - Defense_player_distance[('Y','min')]
    Offense_player_distance['offense_y_spread'] = Offense_player_distance[('Y','Offense_max')] - Offense_player_distance[('Y','Offense_min')]
    player_distance = pd.merge(Offense_player_distance, Defense_player_distance, on=['GameId', 'PlayId'], how='inner')
    train =  pd.merge(train, player_distance, on=['GameId', 'PlayId'], how='inner')

    #closest defense player
    closest_defense_player = train[(train['IsOnOffense'] == False) & (train[('dist_to_rusher', 'min')] == train['dist_to_rusher'])]
    closest_defense_player = closest_defense_player[['GameId', 'PlayId', 'S','A', 'Dir_std', 'Ori_std', 'Dis']].rename(columns={
        'S': 'closest_S','A':'closest_A','Dir_std':'closest_Dir', 'Ori_std':'closest_Ord', 'Dis':'closest_Dis'})
    train =  pd.merge(train, closest_defense_player, on=['GameId', 'PlayId'], how='inner')
    train['rusher_S_closet'] = train['S'] / train['closest_S']
    train['rusher_A_closet'] = train['A'] / train['closest_A']
    train.drop(['closest_S', 'closest_A'],axis = 1, inplace=True)

    # 球员距发球线的距离
    train['dist_to_yardline'] = train[['X','YardLine']].apply(lambda x: x[0] - x[1], axis=1)

    train['Team'] = train['Team'].apply(lambda x: x.strip() == 'home')

    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]
    # train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    # ——————— player ———————
    ## Age
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60 * 60 * 24 * 365.25
    train['PlayerAge'] = train.apply(
        lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
    # train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object") # 是否要将其看成cat变量

    ## Height
    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
    train['PlayerBMI'] = 703 * (train['PlayerWeight'] / (train['PlayerHeight']) ** 2)
    print(2)
    print(train.shape)
    ## Orientation and Dir
    # train["Orientation_ob"] = train["Orientation"].apply(lambda x: orientation_to_cat(x)).astype("object")
    # train["Dir_ob"] = train["Dir"].apply(lambda x: orientation_to_cat(x)).astype("object")
    #
    # train["Orientation_sin"] = train["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    # train["Orientation_cos"] = train["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
    # train["Dir_sin"] = train["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    # train["Dir_cos"] = train["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))

    ## OffensePersonnel
    temp = train[train['IsRusher'] == True]["OffensePersonnel"].apply(lambda x: pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"]
    train = train.merge(temp, on="PlayId")

    ## DefensePersonnel
    temp = train[train['IsRusher'] == True]["DefensePersonnel"].apply(
        lambda x: pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"]
    train = train.merge(temp, on="PlayId")
    print(2.1)
    print(train.shape)

    # train = pd.concat(
    #     [train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')],
    #     axis=1)

    # ——————— environment ———————

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

    ## WindSpeed
    # print(train['WindSpeed'].value_counts())
    # train['WindSpeed_ob'] = train['WindSpeed'].apply(
    #     lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    # train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(
    #     lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)
    # train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(
    #     lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(x) and type(
    #         x) != float and 'gusts up to' in x else x)
    # train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)

    ## Turf
    # Turf = {'Field Turf': 'Artificial', 'A-Turf Titan': 'Artificial', 'Grass': 'Natural',
    #         'UBU Sports Speed S5-M': 'Artificial', 'Artificial': 'Artificial', 'DD GrassMaster': 'Artificial',
    #         'Natural Grass': 'Natural', 'UBU Speed Series-S5-M': 'Artificial', 'FieldTurf': 'Artificial',
    #         'FieldTurf 360': 'Artificial', 'Natural grass': 'Natural', 'grass': 'Natural', 'Natural': 'Natural',
    #         'Artifical': 'Artificial', 'FieldTurf360': 'Artificial', 'Naturall Grass': 'Natural',
    #         'Field turf': 'Artificial', 'SISGrass': 'Artificial', 'Twenty-Four/Seven Turf': 'Artificial',
    #         'natural grass': 'Natural'}
    # train['Turf'] = train['Turf'].map(Turf)
    grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
    train['Grass'] = np.where(train.Turf.str.lower().isin(grass_labels), 1, 0)

    # StadiumType
    train['StadiumType'] = train['StadiumType'].apply(clean_StadiumType)
    train['StadiumType'] = train['StadiumType'].apply(transform_StadiumType)

    # ——————— after possess ———————
    print(3)
    print(train.shape)

    ## sort
    # train = train.sort_values(by=['X']).sort_values(by=['Dis']).sort_values(by=['PlayId']).reset_index(drop=True)
    # train = train.sort_values(by=['X']).sort_values(by=['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop=True)
    # pd.to_pickle(train, "train.pkl")

    train.fillna(-999, inplace=True)

    ## dense -> categorical
    train["Quarter"] = train["Quarter"].astype("object")
    train["Down"] = train["Down"].astype("object")
    train["JerseyNumber"] = train["JerseyNumber"].astype("object")
    train["OffenseFormation"] = train["OffenseFormation"].astype("object")
    # train["YardLine_ob"] = train["YardLine"].astype("object")
    # train["DefendersInTheBox_ob"] = train["DefendersInTheBox"].astype("object")
    train["Week"] = train["Week"].astype("object")
    # train["TimeDelta_ob"] = train["TimeDelta"].astype("object")
    # train["HomeTeamAbbr"] = train["HomeTeamAbbr"].astype("object")
    # train["VisitorTeamAbbr"] = train["VisitorTeamAbbr"].astype("object")

    train = train[train['IsRusher'] == True]  # 树模型中目前只处理rusher

    print(4)
    print(train.shape)

    drop(train)

    print("feature process end,with feature shape:",train.shape)
    # print(train)

    return train


# from kaggle.competitions import nflrush
# env = nflrush.make_env()

if __name__ == '__main__':

    # train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)
    train = pd.read_csv('data/train.csv')[:6600]
    train = preprocess(train)
    # drop后 1358 -> 1276，但LB分变低了，感觉还是不能drop
    # train.drop(train.index[(train['dist_to_end_train'] < train['Yards']) | (train['dist_to_end_train'] - 100 > train['Yards'])],inplace=True)

    y_train = train.pop("Yards")
    X_train = train
    cat_features = []
    dense_features = []
    sss = {}
    for f in X_train.columns:
        if X_train[f].dtype == 'object':
            # print(f)
            cat_features.append((f, len(train[f].unique())))
            lbl = LabelEncoder()
            lbl.fit(list(X_train[f]) + [-999])
            X_train[f] = lbl.transform(list(X_train[f]))
        else:
            ss = StandardScaler()
            X_train.loc[:, f] = ss.fit_transform(X_train[f].values[:, None])
            dense_features.append(f)
            sss[f] = ss

    cdf = get_cdf_df(y_train).values.reshape(-1, )

    kf = KFold(n_splits=5)
    count = 0
    resu1 = 0
    impor1 = 0
    resu2_cprs = 0
    resu3_mae = 0
    ##y_pred = 0
    stack_train = np.zeros([X_train.shape[0], ])
    models = []
    for train_index, test_index in kf.split(X_train, y_train):
        X_train2 = X_train.iloc[train_index, :]
        y_train2 = y_train.iloc[train_index]
        X_test2 = X_train.iloc[test_index, :]
        y_test2 = y_train.iloc[test_index]
        #     clf = lgb.LGBMRegressor(n_estimators=10000, random_state=47,subsample=0.7,
        #                              colsample_bytree=0.7,learning_rate=0.005,importance_type = 'gain',
        #                      max_depth = -1, num_leaves = 100,min_child_samples=20,min_split_gain = 0.001,
        #                        bagging_freq=1,reg_alpha = 0,reg_lambda = 0,n_jobs = -1)
        clf = lgb.LGBMRegressor(n_estimators=10000, random_state=0
                                , learning_rate=0.005, importance_type='gain',
                                n_jobs=-1, metric='mae')
        clf.fit(X_train2, y_train2, eval_set=[(X_train2, y_train2), (X_test2, y_test2)], early_stopping_rounds=200,
                verbose=False)
        models.append(clf)

        ## plot feature importance
        fscores = pd.Series(clf.feature_importances_, X_train2.columns).sort_values(ascending=False)[:20]
        fscores.plot(kind='bar', title='Feature Importance %d' % count, figsize=(20, 10))
        count += 1
        plt.ylabel('Feature Importance Score')
        plt.show()

        temp_predict = clf.predict(X_test2)
        stack_train[test_index] = temp_predict
        ##y_pred += clf.predict(X_test)/5
        mse = mean_squared_error(y_test2, temp_predict)
        crps = CRPS_pingyi1(temp_predict, y_test2, 4, cdf, train['dist_to_end_train'].iloc[test_index])
        mae = mean_absolute_error(y_test2, temp_predict)
        print(crps)

        resu1 += mse / 5
        resu2_cprs += crps / 5
        resu3_mae += mae / 5
        impor1 += clf.feature_importances_ / 5
        gc.collect()
    print('mean mse:', resu1)
    print('oof mse:', mean_squared_error(y_train, stack_train))
    print('mean mae:', resu3_mae)
    print('oof mae:', mean_absolute_error(y_train, stack_train))
    print('mean cprs:', resu2_cprs)
    print('oof cprs:', CRPS_pingyi1(stack_train, y_train, 4, cdf, train['dist_to_end_train']))

    # for (test_df, sample_prediction_df) in env.iter_test():
    #     X_test = preprocess(test_df)
    #     X_test.fillna(-999, inplace=True)
    #     for f in X_test.columns:
    #         if X_test[f].dtype == 'object':
    #             X_test[f] = X_test[f].map(lambda x: x if x in set(X_train[f]) else -999)
    #     for f in X_test.columns:
    #         if X_test[f].dtype == 'object':
    #             lbl = preprocessing.LabelEncoder()
    #             lbl.fit(list(X_train[f]) + [-999])
    #             X_test[f] = lbl.transform(list(X_test[f]))
    #     pred_value = 0
    #     for model in models:
    #         pred_value += model.predict(X_test)[0] / 5
    #     pred_data = list(get_score(pred_value, cdf, 4, X_test['dist_to_end_train'].values[0]))
    #     pred_data = np.array(pred_data).reshape(1, 199)
    #     pred_target = pd.DataFrame(index=sample_prediction_df.index, columns=sample_prediction_df.columns,
    #                                # data = np.array(pred_data))
    #                                data=pred_data)
    #     # print(pred_target)
    #     env.predict(pred_target)
    # env.write_submission_file()

'''
[23171 rows x 56 columns]>
0.013043294731681657
0.012794710635750156
0.012592832061690951
0.014676284850886654
0.014547147048300388
mean mse: 38.846426675027004
oof mse: 38.8464390302977
mean mae: 3.6150459575447407
oof mae: 3.615037035308653
mean cprs: 0.013530853865661961
oof cprs: 0.013530832823879818
'''
