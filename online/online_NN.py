import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import warnings
from string import punctuation
import re
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
import math

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
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
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'other': 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        if dic.__contains__(xxs[-1]):
            dic[xxs[-1]] = int(xxs[-2])
        else:
            dic['other'] = int(xxs[-2])
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
    return (np.sqrt(v * v + 2 * a * dist) - v) / a


def drop(train):
    # drop_cols += ["Orientation", "Dir"]

    play_drop = ["GameId", 'PlayId', "TimeHandoff", "TimeSnap", "GameClock", "DefensePersonnel", "OffensePersonnel",
                 'FieldPosition', 'PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'TeamOnOffense', 'Stadium']
    player_drop = ['DisplayName', 'PlayerBirthDate', "IsRusher", "NflId", "NflIdRusher", "Dir",
                   'Dir_rad', 'Ori_rad', "PlayDirection", 'Orientation', 'Rusher_X', 'Rusher_Y',
                   'dist_to_rusher', 'time_to_rusher']
    environment_drop = ["WindSpeed", "WindDirection", "Season", "GameWeather", 'Location', 'GameWeather_process',
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
    train['dist_yardline'] = train['YardLine'] - train['X'] / 0.91

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
    train['time_to_rusher'] = train[['X', 'Y', 'Rusher_X', 'Rusher_Y', 'S', 'A', ]].apply(
        lambda x: min_tackle_time(euclidean_distance(x[0], x[1], x[2], x[3]), x[4], x[5]), axis=1)
    # train['time_to_rusher_Defend'] = train[train['IsOnOffense'] == False][['X', 'Y','Rusher_X', 'Rusher_Y', 'S', 'A',]].apply(
    #         lambda x: min_tackle_time(euclidean_distance(x[0], x[1], x[2], x[3]), x[4], x[5]), axis=1)

    # Rusher距QB的距离，训练集23171 中有23290个QB，待确认缺失数据的处理
    QB_distance = train[train['Position'] == 'QB'][['dist_to_rusher', 'GameId', 'PlayId']].rename(
        columns={'dist_to_rusher': 'dist_QB'})
    train = pd.merge(train, QB_distance, on=['GameId', 'PlayId'], how='left')
    # print("0.3", train.shape)

    # let's say now I want for that specific play to have as features the # of players within 3, 6, 9, 12, 15
    # yards of distance from the runner. In that case, as I already have the distances from the runner to each of the 11 defense players, I will count how many of them are within each of these intervals, and return those.
    # defense_x = train[train['IsOnOffense'] == False][['X','Rusher_X','GameId','PlayId']].apply

    # 每个play对应两条,敌方球员距离，友方球员距离
    Offense_player_distance = train[(train['IsOnOffense'] == True) & (train['dist_to_rusher'] > 0)].groupby(
        ['GameId', 'PlayId']) \
        .agg({'dist_to_rusher': ['min', 'max', 'mean', 'std'],
              'X': ['mean', 'std'], 'Y': ['max', 'min', 'mean', 'std'],
              'time_to_rusher': ['mean', 'min']
              }).rename(columns={
        'min': 'Offense_min', 'max': 'Offense_max', 'mean': 'Offense_mean', 'std': 'Offense_std'}).reset_index()
    Defense_player_distance = train[train['IsOnOffense'] == False].groupby(['GameId', 'PlayId']) \
        .agg({'dist_to_rusher': ['min', 'max', 'mean', 'std'],
              'X': ['mean', 'std'], 'Y': ['mean', 'std', 'max', 'min'],
              'time_to_rusher': ['mean', 'min']
              }).reset_index()  # min表示防守方跑的最快的球员跑到rusher的时间
    player_distance = pd.merge(Offense_player_distance, Defense_player_distance, on=['GameId', 'PlayId'], how='left')
    train = pd.merge(train, player_distance, on=['GameId', 'PlayId'], how='left')
    train['defense_y_spread'] = train[('Y', 'max')] - train[('Y', 'min')]
    train['offense_y_spread'] = train[('Y', 'Offense_max')] - train[('Y', 'Offense_min')]

    # closest defense player
    closest_defense_player = train[
        (train['IsOnOffense'] == False) & (train[('dist_to_rusher', 'min')] == train['dist_to_rusher'])]
    closest_defense_player = closest_defense_player[['GameId', 'PlayId', 'S', 'A', 'Dir_std', 'Ori_std', 'Dis']].rename(
        columns={
            'S': 'closest_S', 'A': 'closest_A', 'Dir_std': 'closest_Dir', 'Ori_std': 'closest_Ord',
            'Dis': 'closest_Dis'})
    train = pd.merge(train, closest_defense_player, on=['GameId', 'PlayId'], how='left')
    train['rusher_S_closet'] = train['S'] / (train['closest_S'] + 0.001)
    train['rusher_A_closet'] = train['A'] / (train['closest_A'] + 0.001)
    train.drop(['closest_S', 'closest_A'], axis=1, inplace=True)

    # 球员距发球线的距离
    train['dist_to_yardline'] = train[['X', 'YardLine']].apply(lambda x: x[0] - x[1], axis=1)

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
    print(train.shape)

    ## OffensePersonnel
    temp = train[train['IsRusher'] == True]["OffensePersonnel"].apply(
        lambda x: pd.Series(OffensePersonnelSplit(x))).reset_index()
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"]
    print(temp.shape)
    train = train.merge(temp, on="PlayId", how='left')
    print(train.shape)
    ## DefensePersonnel
    temp = train[train['IsRusher'] == True]["DefensePersonnel"].apply(
        lambda x: pd.Series(DefensePersonnelSplit(x))).reset_index()
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"]
    train = train.merge(temp, on="PlayId", how='left')

    print(4)
    print(train.shape)

    drop(train)

    # train.fillna(-999, inplace=True)

    print("feature process end,with feature shape:", train.shape)
    # print(train)

    return train


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train.shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


class GP:
    def __init__(self):
        self.classes = 20
        self.class_names = ['class_0',
                            'class_1',
                            'class_2',
                            'class_3',
                            'class_4',
                            'class_5',
                            'class_6',
                            'class_7',
                            'class_8',
                            'class_9',
                            'class_10',
                            'class_11',
                            'class_12',
                            'class_13',
                            'class_14',
                            'class_15',
                            'class_16',
                            'class_17',
                            'class_18',
                            'class_19',
                            ]

    def GrabPredictions(self, data):
        oof_preds = np.zeros((len(data), len(self.class_names)))
        oof_preds[:, 0] = self.GP_class_0(data)
        oof_preds[:, 1] = self.GP_class_1(data)
        oof_preds[:, 2] = self.GP_class_2(data)
        oof_preds[:, 3] = self.GP_class_3(data)
        oof_preds[:, 4] = self.GP_class_4(data)
        oof_preds[:, 5] = self.GP_class_5(data)
        oof_preds[:, 6] = self.GP_class_6(data)
        oof_preds[:, 7] = self.GP_class_7(data)
        oof_preds[:, 8] = self.GP_class_8(data)
        oof_preds[:, 9] = self.GP_class_9(data)
        oof_preds[:, 10] = self.GP_class_10(data)
        oof_preds[:, 11] = self.GP_class_11(data)
        oof_preds[:, 12] = self.GP_class_12(data)
        oof_preds[:, 13] = self.GP_class_13(data)
        oof_preds[:, 14] = self.GP_class_14(data)
        oof_preds[:, 15] = self.GP_class_15(data)
        oof_preds[:, 16] = self.GP_class_16(data)
        oof_preds[:, 17] = self.GP_class_17(data)
        oof_preds[:, 18] = self.GP_class_18(data)
        oof_preds[:, 19] = self.GP_class_19(data)
        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)
        oof_df = oof_df.div(oof_df.sum(axis=1), axis=0)

        return oof_df.values

    def GP_class_0(self, data):
        return (0.250000 * np.tanh(((((((data[:, 0]) - (((data[:, 0]) + (data[:, 0]))))) + (data[:, 0]))) + (
            ((((data[:, 0]) * 2.0)) * 2.0)))) +
                0.250000 * np.tanh((((
                        (((((((((data[:, 2]) - (data[:, 7]))) * 2.0)) + (data[:, 7]))) * 2.0)) + (data[:, 0]))) + (
                                        ((((data[:, 2]) + (data[:, 0]))) - (data[:, 7]))))) +
                0.250000 * np.tanh((((((((((((
                        (((((((data[:, 3]) - (data[:, 7]))) - (((data[:, 0]) / 2.0)))) + (data[:, 0]))) - (
                    data[:, 7]))) + (data[:, 2]))) + (((data[:, 7]) + (data[:, 2]))))) + (data[:, 7]))) - (
                                          data[:, 7]))) + (data[:, 0]))) +
                0.250000 * np.tanh(((((((((data[:, 0]) - (data[:, 7]))) - (data[:, 14]))) + (
                    ((data[:, 17]) - (data[:, 14]))))) + (
                                        ((((((data[:, 0]) - ((1.0)))) - (data[:, 0]))) + (data[:, 20]))))) +
                0.250000 * np.tanh((((
                        (data[:, 18]) + (((((((data[:, 0]) - (data[:, 7]))) + (data[:, 2]))) + (data[:, 0]))))) - (
                                        ((data[:, 7]) + (data[:, 2]))))) +
                0.250000 * np.tanh((((((((data[:, 8]) + ((
                        ((((((data[:, 13]) + (data[:, 3]))) + (data[:, 0])) / 2.0)) + (((((((
                        (data[:, 0]) + (((((data[:, 0]) - (data[:, 0]))) / 2.0)))) + (data[:, 17])) / 2.0)) + (
                                                                                            data[:,
                                                                                            11])))))) / 2.0)) + (
                                          data[:, 0]))) - (data[:, 13]))) +
                0.250000 * np.tanh(((((data[:, 0]) * 2.0)) + (((((((((data[:, 2]) + (data[:, 2]))) * 2.0)) - (
                    ((data[:, 14]) - (((np.tanh((data[:, 2]))) * 2.0)))))) - (data[:, 7]))))) +
                0.250000 * np.tanh((((((((((((
                        (data[:, 17]) - (((data[:, 17]) * (((((data[:, 17]) * (data[:, 17]))) * 2.0)))))) + (
                                                ((((((data[:, 17]) * 2.0)) * 2.0)) * 2.0))) / 2.0)) / 2.0)) + (
                                           ((data[:, 17]) - ((7.76236486434936523))))) / 2.0)) + (
                                        ((((((data[:, 17]) * (data[:, 17]))) * 2.0)) * 2.0)))) +
                0.250000 * np.tanh(((((((((((data[:, 17]) + (data[:, 17])) / 2.0)) + (
                    (((data[:, 17]) + ((((data[:, 17]) + (data[:, 17])) / 2.0))) / 2.0))) / 2.0)) * (data[:, 17]))) - (
                                        data[:, 13]))) +
                0.250000 * np.tanh((((((data[:, 19]) + ((
                        (((((((data[:, 19]) * 2.0)) + (data[:, 19]))) - ((1.07645297050476074)))) + (
                    ((data[:, 22]) + (data[:, 22])))))) / 2.0)) + (((data[:, 22]) + (data[:, 4]))))))

    def GP_class_1(self, data):
        return (0.250000 * np.tanh((((((((
                (((((data[:, 3]) + (((np.tanh((data[:, 2]))) - (data[:, 14]))))) - ((10.0)))) - (
            (4.88989353179931641)))) - ((13.92175483703613281)))) / 2.0)) - (data[:, 6]))) +
                0.250000 * np.tanh(data[:, 3]) +
                0.250000 * np.tanh(((data[:, 2]) + (((((np.tanh((((data[:, 14]) * 2.0)))) - (
                    ((((((data[:, 7]) * 2.0)) - (((((data[:, 0]) - (data[:, 14]))) - (data[:, 7]))))) - (
                        data[:, 0]))))) + (
                                                         data[:, 0]))))) +
                0.250000 * np.tanh(((((((data[:, 0]) - ((-1.0 * ((data[:, 14])))))) + (((((((
                        (((data[:, 0]) - (data[:, 7]))) - ((((data[:, 14]) + (data[:, 14])) / 2.0)))) + ((
                        (data[:, 2]) - (((data[:, 0]) * (data[:, 14])))))) / 2.0)) - (data[:, 14]))))) - (
                                        data[:, 7]))) +
                0.250000 * np.tanh(((((data[:, 0]) / 2.0)) + (data[:, 0]))) +
                0.250000 * np.tanh(((((data[:, 19]) - (data[:, 19]))) / 2.0)) +
                0.250000 * np.tanh(((((np.tanh((((np.tanh((data[:, 5]))) - ((((1.69341480731964111)) + (
                    (((((np.tanh((data[:, 14]))) - (data[:, 14]))) + (data[:, 14])) / 2.0)))))))) - (data[:, 2]))) - (
                                        ((data[:, 14]) * 2.0)))) +
                0.250000 * np.tanh((((((data[:, 22]) + (((data[:, 0]) + (
                    (((((((data[:, 22]) / 2.0)) + ((-1.0 * ((data[:, 13]))))) / 2.0)) + (data[:, 19])))))) / 2.0)) - (
                                        data[:, 13]))) +
                0.250000 * np.tanh(((((((data[:, 17]) / 2.0)) * (data[:, 17]))) * (((data[:, 17]) + (((data[:, 17]) * ((
                        (((((((data[:, 17]) + (data[:, 17]))) * (data[:, 17]))) * (data[:, 17]))) + (
                    ((np.tanh((data[:, 17]))) / 2.0)))))))))) +
                0.250000 * np.tanh((((((data[:, 14]) * (((data[:, 13]) + (data[:, 3]))))) + (
                    ((data[:, 14]) * (((data[:, 14]) - ((-1.0 * ((data[:, 3]))))))))) / 2.0)))

    def GP_class_2(self, data):
        return (0.250000 * np.tanh(((data[:, 0]) + (np.tanh(
            (((((((((data[:, 2]) + (((data[:, 2]) / 2.0))) / 2.0)) + (data[:, 0])) / 2.0)) * (data[:, 7]))))))) +
                0.250000 * np.tanh((((data[:, 22]) + (data[:, 22])) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 14]) + ((-1.0 * ((((((4.73206138610839844)) + (((((((
                        ((((np.tanh((data[:, 7]))) + (data[:, 1])) / 2.0)) - (
                    np.tanh((((((data[:, 0]) - (data[:, 20]))) - (data[:, 14]))))))) * ((7.0)))) + (data[:,
                                                                                                    14])) / 2.0))) / 2.0)))))) / 2.0)) - (
                                        data[:, 14]))) +
                0.250000 * np.tanh(((((data[:, 1]) + ((4.15603733062744141)))) / 2.0)) +
                0.250000 * np.tanh(((((((((((((data[:, 0]) - (data[:, 7]))) - (data[:, 7]))) + (
                    (((data[:, 0]) + (data[:, 0])) / 2.0)))) * 2.0)) + (data[:, 0]))) * 2.0)) +
                0.250000 * np.tanh(((((data[:, 11]) / 2.0)) / 2.0)) +
                0.250000 * np.tanh(((((((((((((
                        ((((((((((data[:, 2]) / 2.0)) + (((data[:, 2]) / 2.0))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) + ((
                        ((-1.0 * ((((data[:, 0]) / 2.0))))) * (((data[:, 0]) / 2.0))))) / 2.0)) + (
                                              data[:, 0])) / 2.0)) + (np.tanh((data[:, 0])))) / 2.0)) / 2.0)) +
                0.250000 * np.tanh(((((data[:, 0]) + (
                    ((data[:, 0]) + (((((((data[:, 14]) * (data[:, 14]))) + (data[:, 14]))) - (data[:, 0]))))))) - (
                                        data[:, 14]))) +
                0.250000 * np.tanh((((((np.tanh((np.tanh(((0.0)))))) * (data[:, 15]))) + (data[:, 20])) / 2.0)) +
                0.250000 * np.tanh(((data[:, 14]) * ((((data[:, 13]) + ((((data[:, 13]) + ((((data[:, 14]) + (((((
                        (data[:, 14]) + ((((data[:, 14]) + (
                    ((data[:, 13]) - ((((-1.0 * (((((data[:, 14]) + (data[:, 14])) / 2.0))))) * 2.0))))) / 2.0)))) + (
                                                                                                                    data[
                                                                                                                    :,
                                                                                                                    14])) / 2.0))) / 2.0))) / 2.0))) / 2.0)))))

    def GP_class_3(self, data):
        return (0.250000 * np.tanh(((((
                ((((8.0)) + (((((((5.33416414260864258)) + ((9.0))) / 2.0)) + ((((8.0)) * 2.0)))))) * 2.0)) + (
                                         (5.33416414260864258))) / 2.0)) +
                0.250000 * np.tanh((((9.48088836669921875)) + (
                    ((((10.56953334808349609)) + (
                        (((4.48959350585937500)) + (np.tanh(((((3.0)) + ((9.0))))))))) / 2.0)))) +
                0.250000 * np.tanh((((((((((data[:, 22]) / 2.0)) + (data[:, 22])) / 2.0)) * 2.0)) / 2.0)) +
                0.250000 * np.tanh((10.44883441925048828)) +
                0.250000 * np.tanh(((((data[:, 0]) + (((data[:, 0]) + ((-1.0 * ((data[:, 9])))))))) + (
                    ((((data[:, 0]) - (data[:, 7]))) - (data[:, 9]))))) +
                0.250000 * np.tanh(((np.tanh((data[:, 6]))) - (data[:, 21]))) +
                0.250000 * np.tanh(((data[:, 0]) - (((((data[:, 14]) + ((
                        (((data[:, 14]) + (((data[:, 14]) + (((data[:, 14]) - (data[:, 0]))))))) - (
                    ((((data[:, 0]) - (data[:, 14]))) + (data[:, 14]))))))) - (
                                                         ((((data[:, 14]) * (data[:, 14]))) + (data[:, 18]))))))) +
                0.250000 * np.tanh(((data[:, 14]) * ((((data[:, 13]) + (data[:, 14])) / 2.0)))) +
                0.250000 * np.tanh((((((data[:, 8]) * (((data[:, 8]) / 2.0)))) + (
                    (((((data[:, 9]) * (data[:, 8]))) + (data[:, 8])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh((((data[:, 2]) + (((((((data[:, 2]) + (data[:, 20])) / 2.0)) + (
                    (((((((data[:, 2]) + (data[:, 8])) / 2.0)) / 2.0)) - (data[:, 8])))) / 2.0))) / 2.0)))

    def GP_class_4(self, data):
        return (0.250000 * np.tanh((((12.98819637298583984)) / 2.0)) +
                0.250000 * np.tanh(((((4.75780344009399414)) + (
                    ((((((8.66014671325683594)) + ((4.18107128143310547))) / 2.0)) * (
                        (8.97841739654541016))))) / 2.0)) +
                0.250000 * np.tanh((((9.0)) / 2.0)) +
                0.250000 * np.tanh(((((((
                        ((((10.0)) + (np.tanh((((((6.0)) + ((((13.80149555206298828)) + ((7.0))))) / 2.0)))))) + (
                    (10.0)))) + (((((((data[:, 0]) + ((8.0)))) / 2.0)) / 2.0))) / 2.0)) + (
                                        (((-1.0 * ((((data[:, 2]) * (data[:, 9])))))) * 2.0)))) +
                0.250000 * np.tanh((((
                        (data[:, 0]) + (((((((data[:, 22]) + (data[:, 22])) / 2.0)) + (data[:, 22])) / 2.0)))) + (((
                                                                                                                           (
                                                                                                                               (
                                                                                                                                       (
                                                                                                                                           data[
                                                                                                                                           :,
                                                                                                                                           22]) + (
                                                                                                                                           (
                                                                                                                                                   (
                                                                                                                                                           (
                                                                                                                                                               (
                                                                                                                                                                       (
                                                                                                                                                                           data[
                                                                                                                                                                           :,
                                                                                                                                                                           22]) * 2.0)) + (
                                                                                                                                                               data[
                                                                                                                                                               :,
                                                                                                                                                               22])) / 2.0)))) + (
                                                                                                                               data[
                                                                                                                               :,
                                                                                                                               18])) / 2.0)))) +
                0.250000 * np.tanh(((((((((
                        ((((np.tanh((data[:, 18]))) + ((((data[:, 1]) + (data[:, 18])) / 2.0))) / 2.0)) * (
                    data[:, 18]))) * (data[:, 18]))) + (((data[:, 18]) * (data[:, 18])))) / 2.0)) - (((((
                        ((np.tanh((data[:, 18]))) + (((data[:, 18]) * (data[:, 18])))) / 2.0)) + (data[:,
                                                                                                  17])) / 2.0)))) +
                0.250000 * np.tanh(((np.tanh((((((
                        (np.tanh((data[:, 6]))) - (((np.tanh((data[:, 6]))) - (data[:, 6]))))) + (
                                                    np.tanh((data[:, 6])))) / 2.0)))) - ((((data[:, 10]) + ((((data[:,
                                                                                                               2]) + ((
                        (np.tanh((data[:, 6]))) - (
                    ((np.tanh((np.tanh((data[:, 6]))))) - (data[:, 10])))))) / 2.0))) / 2.0)))) +
                0.250000 * np.tanh((((((((((((data[:, 10]) + (data[:, 4]))) + (
                    ((data[:, 10]) + (((data[:, 0]) * (data[:, 10])))))) / 2.0)) + (
                                            ((data[:, 4]) * (((data[:, 10]) + (data[:, 4])))))) / 2.0)) + (
                                         data[:, 7])) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 11]) + (
                    ((((data[:, 18]) / 2.0)) * (((data[:, 11]) * (data[:, 18])))))) / 2.0)) * (((data[:, 18]) * ((((((((
                        (data[:, 18]) * (((((((data[:, 11]) / 2.0)) / 2.0)) * (data[:, 18]))))) / 2.0)) / 2.0)) * ((
                        -1.0 * ((data[:, 21])))))))))) +
                0.250000 * np.tanh((((-1.0 * ((((((((((((data[:, 2]) - (data[:, 2]))) * (np.tanh((data[:, 2]))))) + (
                    np.tanh((((data[:, 14]) * 2.0))))) / 2.0)) + (data[:, 2])) / 2.0))))) - (data[:, 2]))))

    def GP_class_5(self, data):
        return (0.250000 * np.tanh(((((((((((3.46574378013610840)) + (
            (((np.tanh(((3.46574378013610840)))) + ((8.46705245971679688))) / 2.0))) / 2.0)) * 2.0)) + (
                                          (((3.46574378013610840)) + ((4.0)))))) + ((((4.0)) * 2.0)))) +
                0.250000 * np.tanh((((10.55856513977050781)) / 2.0)) +
                0.250000 * np.tanh((((3.76695251464843750)) / 2.0)) +
                0.250000 * np.tanh((((((6.0)) / 2.0)) + ((((8.57984828948974609)) * 2.0)))) +
                0.250000 * np.tanh((((((((((((3.42168045043945312)) + (data[:, 7]))) + (np.tanh((data[:, 7]))))) + (
                    np.tanh((np.tanh((np.tanh(((3.42168045043945312))))))))) / 2.0)) + (data[:, 7])) / 2.0)) +
                0.250000 * np.tanh(((((((((((data[:, 6]) - (data[:, 9]))) + (data[:, 7])) / 2.0)) * 2.0)) + (
                    (((((data[:, 7]) + (data[:, 9])) / 2.0)) - (data[:, 9])))) / 2.0)) +
                0.250000 * np.tanh(
                    ((((((((((data[:, 22]) + (data[:, 22]))) + (data[:, 11])) / 2.0)) + (data[:, 22])) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((((((((0.76044219732284546)) / 2.0)) + ((0.76044219732284546))) / 2.0)) + (
                    np.tanh((np.tanh((((((((0.76043862104415894)) / 2.0)) + ((1.0))) / 2.0))))))) / 2.0)) +
                0.250000 * np.tanh(((((((data[:, 12]) + (np.tanh((data[:, 9])))) / 2.0)) + (((((((((((data[:, 12]) + (((
                                                                                                                               (
                                                                                                                                   (
                                                                                                                                       0.0)) + (
                                                                                                                                   np.tanh(
                                                                                                                                       (
                                                                                                                                           (
                                                                                                                                                   (
                                                                                                                                                           (
                                                                                                                                                               (
                                                                                                                                                                       (
                                                                                                                                                                               (
                                                                                                                                                                                   data[
                                                                                                                                                                                   :,
                                                                                                                                                                                   7]) + (
                                                                                                                                                                                   data[
                                                                                                                                                                                   :,
                                                                                                                                                                                   6])) / 2.0)) + (
                                                                                                                                                               data[
                                                                                                                                                               :,
                                                                                                                                                               7])) / 2.0))))) / 2.0))) / 2.0)) / 2.0)) / 2.0)) + (
                                                                                                  data[:,
                                                                                                  7])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((((((data[:, 20]) / 2.0)) / 2.0)) * (
                    ((np.tanh((data[:, 20]))) * (((data[:, 20]) / 2.0)))))) * (
                                        (((((data[:, 20]) / 2.0)) + (((data[:, 20]) / 2.0))) / 2.0)))))

    def GP_class_6(self, data):
        return (0.250000 * np.tanh(((((((11.68076229095458984)) + (((((11.68075847625732422)) + ((
                ((11.47270107269287109)) + (
            ((((11.47269725799560547)) + ((11.68076229095458984))) / 2.0))))) / 2.0))) / 2.0)) + ((((4.0)) * 2.0)))) +
                0.250000 * np.tanh((((8.0)) + ((((10.0)) + ((6.59042119979858398)))))) +
                0.250000 * np.tanh(((((((((((13.33760547637939453)) + (((((3.86388397216796875)) + (((((
                        ((8.18321037292480469)) + (data[:, 4]))) + (((((
                        ((5.95386552810668945)) + ((12.93883609771728516)))) + ((
                        (((10.73907089233398438)) + ((4.0))) / 2.0))) / 2.0))) / 2.0))) / 2.0))) / 2.0)) + (
                                            np.tanh(((3.0)))))) - ((2.0)))) + (
                                        ((((10.61559963226318359)) + (data[:, 1])) / 2.0)))) +
                0.250000 * np.tanh(((np.tanh(((((((((((7.13513946533203125)) + ((12.48778533935546875))) / 2.0)) + (((((
                        ((7.13513565063476562)) - ((8.12031841278076172)))) + (((((
                        ((((7.13513565063476562)) + ((((10.69830417633056641)) + ((10.69830799102783203)))))) + (
                    (10.69830799102783203)))) + ((7.83078956604003906))) / 2.0))) / 2.0))) / 2.0)) * 2.0)))) + (
                                        (3.0)))) +
                0.250000 * np.tanh(
                    ((((((((((data[:, 7]) - (data[:, 0]))) * 2.0)) - (data[:, 7]))) + (data[:, 7]))) + (data[:, 0]))) +
                0.250000 * np.tanh(((((data[:, 7]) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((((((((((((((1.0)) + (((((
                        ((data[:, 2]) + (((((((1.0)) * (data[:, 2]))) + (data[:, 14])) / 2.0))) / 2.0)) + (
                                                                 (1.0))) / 2.0)))) + (
                                                  (((1.0)) - (((data[:, 2]) * (data[:, 14])))))) / 2.0)) - (
                                               data[:, 2]))) * 2.0)) + (data[:, 14])) / 2.0)) + (data[:, 14]))) +
                0.250000 * np.tanh(((np.tanh((np.tanh(((((((data[:, 19]) * ((0.0)))) + ((0.0))) / 2.0)))))) / 2.0)) +
                0.250000 * np.tanh((((((((0.0)) / 2.0)) / 2.0)) * (
                    ((((((((data[:, 15]) / 2.0)) * ((0.0)))) / 2.0)) * ((((-1.0 * (((0.0))))) / 2.0)))))) +
                0.250000 * np.tanh(((((data[:, 15]) * (((((((((((((np.tanh(
                    ((((0.09032609313726425)) / 2.0)))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) / 2.0)))

    def GP_class_7(self, data):
        return (0.250000 * np.tanh((10.27210903167724609)) +
                0.250000 * np.tanh((((((((((10.0)) + (
                    ((((((((8.26972770690917969)) + ((11.06334972381591797))) / 2.0)) * 2.0)) / 2.0)))) + (
                                            (((((5.0)) + ((((5.0)) + ((7.92727756500244141)))))) * 2.0)))) * (
                                          (7.92728137969970703)))) + ((10.0)))) +
                0.250000 * np.tanh((8.42519950866699219)) +
                0.250000 * np.tanh(((((((data[:, 14]) + (data[:, 14]))) + (
                    (((((1.59392631053924561)) + (data[:, 14]))) + (data[:, 14]))))) + (np.tanh((((((data[:, 14]) + (
                    ((((((3.0)) + (((data[:, 14]) * 2.0))) / 2.0)) + ((1.59391915798187256)))))) + ((
                    1.59392631053924561)))))))) +
                0.250000 * np.tanh((((data[:, 7]) + (((data[:, 0]) + (
                    ((((((data[:, 7]) + (data[:, 4])) / 2.0)) + (((data[:, 7]) * 2.0))) / 2.0))))) / 2.0)) +
                0.250000 * np.tanh(((((((((
                        (((((data[:, 9]) - (((data[:, 0]) - (data[:, 0]))))) - (((data[:, 0]) / 2.0)))) / 2.0)) - (
                                             ((data[:, 21]) - (data[:, 7]))))) + (data[:, 21])) / 2.0)) - (
                                        data[:, 0]))) +
                0.250000 * np.tanh((((((((((2.0)) + ((((((data[:, 21]) + ((((data[:, 13]) + (((((((
                        ((((data[:, 13]) * 2.0)) + (data[:, 21])) / 2.0)) + (data[:,
                                                                             13])) / 2.0)) * 2.0))) / 2.0))) / 2.0)) - (
                                                          np.tanh(((((0.0)) - (data[:, 21]))))))))) + (
                                            (1.0))) / 2.0)) + (data[:, 13])) / 2.0)) +
                0.250000 * np.tanh(((((data[:, 14]) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((((((((((((
                        ((-1.0 * (((((((data[:, 21]) / 2.0)) + (data[:, 8])) / 2.0))))) / 2.0)) / 2.0)) / 2.0)) + (
                                              data[:, 9])) / 2.0)) / 2.0)) + ((
                        ((((((((((data[:, 7]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) + (data[:, 7])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((((np.tanh(((1.0)))) / 2.0)) / 2.0)) / 2.0)))

    def GP_class_8(self, data):
        return (0.250000 * np.tanh(((((((data[:, 0]) + (((((6.71804094314575195)) + (data[:, 15])) / 2.0)))) + (
            (((((11.56385326385498047)) * 2.0)) * ((((10.48986148834228516)) + ((11.56385326385498047)))))))) / 2.0)) +
                0.250000 * np.tanh((((((((((data[:, 14]) / 2.0)) + ((2.83755254745483398))) / 2.0)) + (
                    (((((data[:, 12]) + (data[:, 14])) / 2.0)) + (data[:, 14]))))) + (
                                        (((data[:, 9]) + (((data[:, 12]) + (data[:, 14])))) / 2.0)))) +
                0.250000 * np.tanh(((((((((data[:, 6]) + (((((
                        ((3.50821948051452637)) + ((((data[:, 21]) + ((11.92932796478271484))) / 2.0)))) + (
                                                                (1.0))) / 2.0)))) + (data[:, 10])) / 2.0)) + (
                                         data[:, 21])) / 2.0)) +
                0.250000 * np.tanh(
                    ((data[:, 14]) + (np.tanh((np.tanh(((-1.0 * ((((data[:, 22]) - (data[:, 13])))))))))))) +
                0.250000 * np.tanh(((((1.0)) + ((((data[:, 7]) + (data[:, 7])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((((((data[:, 21]) / 2.0)) / 2.0)) * ((((
                        (((np.tanh((((np.tanh((data[:, 13]))) / 2.0)))) / 2.0)) + ((((
                        (((np.tanh(((((data[:, 21]) + (data[:, 21])) / 2.0)))) / 2.0)) - (
                    data[:, 12]))) / 2.0)))) / 2.0)))) - (data[:, 12]))) +
                0.250000 * np.tanh((((data[:, 7]) + (np.tanh(((((data[:, 12]) + (
                    ((data[:, 7]) * ((((data[:, 13]) + (data[:, 13])) / 2.0))))) / 2.0))))) / 2.0)) +
                0.250000 * np.tanh(((((((data[:, 13]) / 2.0)) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((((((((np.tanh((data[:, 9]))) + (data[:, 13])) / 2.0)) + (
                    (((((((data[:, 21]) / 2.0)) * 2.0)) + ((((3.94983267784118652)) / 2.0))) / 2.0))) / 2.0)) + (
                                         (((np.tanh(((3.94983267784118652)))) + (data[:, 9])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((((((((((((data[:, 14]) * (((((((((data[:, 19]) / 2.0)) * (
                    np.tanh((((data[:, 2]) / 2.0)))))) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) * (
                                        ((((data[:, 11]) / 2.0)) / 2.0)))))

    def GP_class_9(self, data):
        return (0.250000 * np.tanh(((((((data[:, 14]) * 2.0)) / 2.0)) + (
            ((((data[:, 14]) + (np.tanh((((data[:, 14]) * 2.0)))))) + (data[:, 14]))))) +
                0.250000 * np.tanh(((data[:, 9]) - (((((-1.0 * ((data[:, 11])))) + (data[:, 9])) / 2.0)))) +
                0.250000 * np.tanh(((data[:, 21]) * 2.0)) +
                0.250000 * np.tanh((((((data[:, 14]) + (data[:, 14])) / 2.0)) + (
                    (((data[:, 13]) + ((((np.tanh((((data[:, 9]) + ((6.0)))))) + (data[:, 9])) / 2.0))) / 2.0)))) +
                0.250000 * np.tanh(np.tanh(((1.0)))) +
                0.250000 * np.tanh(((np.tanh((np.tanh(((((np.tanh((((data[:, 20]) * (((((
                        ((np.tanh((((data[:, 20]) / 2.0)))) + (data[:, 12])) / 2.0)) + ((
                        ((data[:, 15]) + ((((data[:, 17]) + (data[:, 9])) / 2.0))) / 2.0))) / 2.0)))))) + (
                                                             data[:, 9])) / 2.0)))))) / 2.0)) +
                0.250000 * np.tanh(((((data[:, 2]) - (data[:, 11]))) / 2.0)) +
                0.250000 * np.tanh((((((((data[:, 13]) / 2.0)) + (((data[:, 7]) + (data[:, 7])))) / 2.0)) - ((((
                        (data[:, 22]) + (((((data[:, 14]) * (((((data[:, 7]) * (((data[:, 13]) - ((((data[:, 7]) + (
                    ((((data[:, 13]) / 2.0)) + (data[:, 7])))) / 2.0)))))) / 2.0)))) * 2.0)))) / 2.0)))) +
                0.250000 * np.tanh(((((((((((((((data[:, 1]) / 2.0)) + ((1.0))) / 2.0)) + ((1.0))) / 2.0)) + (((((
                        (((((1.0)) / 2.0)) + (np.tanh(((1.0))))) / 2.0)) + (np.tanh(
                    (((((1.0)) + ((1.0))) / 2.0))))) / 2.0))) / 2.0)) + ((((data[:, 20]) + ((1.0))) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((data[:, 1]) * ((((data[:, 14]) + (np.tanh(((((0.0)) / 2.0))))) / 2.0)))))

    def GP_class_10(self, data):
        return (0.250000 * np.tanh((((((((((data[:, 2]) + (((data[:, 9]) * (np.tanh((data[:, 14])))))) / 2.0)) + (
            (((((data[:, 14]) + (data[:, 20])) / 2.0)) + (data[:, 0])))) / 2.0)) + (data[:, 14])) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 9]) + ((-1.0 * ((((((((2.04309988021850586)) + (
                    ((data[:, 11]) + (((np.tanh((data[:, 0]))) / 2.0))))) / 2.0)) * 2.0)))))) / 2.0)) + (data[:, 2]))) +
                0.250000 * np.tanh(data[:, 21]) +
                0.250000 * np.tanh(((
                                        (((((data[:, 14]) + (np.tanh((((data[:, 6]) / 2.0))))) / 2.0)) + (
                                            data[:, 14]))) + (
                                        ((((data[:, 14]) * 2.0)) + ((((data[:, 13]) + (data[:, 14])) / 2.0)))))) +
                0.250000 * np.tanh(data[:, 9]) +
                0.250000 * np.tanh(((((((data[:, 15]) * 2.0)) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((((((2.72836518287658691)) * ((((1.0)) / 2.0)))) / 2.0)) / 2.0)) +
                0.250000 * np.tanh(((((np.tanh(
                    (((data[:, 7]) + (np.tanh(((((((((((0.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))))))) / 2.0)) / 2.0)) +
                0.250000 * np.tanh(np.tanh((((np.tanh(((11.43236827850341797)))) / 2.0)))) +
                0.250000 * np.tanh(np.tanh((((data[:, 7]) / 2.0)))))

    def GP_class_11(self, data):
        return (0.250000 * np.tanh((-1.0 * (((((((((((
                (((data[:, 5]) - (((data[:, 9]) - ((((-1.0 * (((11.40053558349609375))))) / 2.0)))))) + (
            (((11.40053558349609375)) + ((((11.40053558349609375)) * 2.0)))))) / 2.0)) - (data[:, 7]))) - (
                                                    (-1.0 * (((11.40053939819335938))))))) * 2.0))))) +
                0.250000 * np.tanh(((((data[:, 14]) * 2.0)) + (data[:, 14]))) +
                0.250000 * np.tanh(((data[:, 2]) + (
                    ((((data[:, 5]) + (data[:, 3]))) + ((((data[:, 2]) + (((data[:, 3]) * 2.0))) / 2.0)))))) +
                0.250000 * np.tanh(((((((((((data[:, 21]) + (data[:, 21]))) + ((-1.0 * (
                    ((((np.tanh((np.tanh((((data[:, 14]) / 2.0)))))) + (data[:, 21])) / 2.0)))))) / 2.0)) * 2.0)) + (
                                         (((data[:, 14]) + (data[:, 3])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((((data[:, 0]) + ((((data[:, 9]) + ((((data[:, 9]) + (((((((
                        ((((((((data[:, 9]) / 2.0)) - (data[:, 10]))) + (data[:, 5])) / 2.0)) - (data[:, 9]))) + (
                                                                                                    data[:,
                                                                                                    9])) / 2.0)) / 2.0))) / 2.0))) / 2.0))) / 2.0)) + (
                                         data[:, 9])) / 2.0)) +
                0.250000 * np.tanh(
                    (((((((data[:, 13]) + ((((data[:, 13]) + (data[:, 13])) / 2.0)))) + (data[:, 1])) / 2.0)) / 2.0)) +
                0.250000 * np.tanh(data[:, 14]) +
                0.250000 * np.tanh(((((data[:, 2]) / 2.0)) / 2.0)) +
                0.250000 * np.tanh((((data[:, 9]) + (data[:, 9])) / 2.0)) +
                0.250000 * np.tanh(data[:, 2]))

    def GP_class_12(self, data):
        return (0.250000 * np.tanh(
            (((((((-1.0 * ((data[:, 18])))) - (np.tanh((data[:, 9]))))) + ((8.0)))) - ((14.74865913391113281)))) +
                0.250000 * np.tanh((((((data[:, 21]) + (np.tanh((data[:, 9]))))) + (
                    (((((((data[:, 5]) / 2.0)) * 2.0)) + (data[:, 5])) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((data[:, 14]) + (data[:, 14]))) + (((data[:, 14]) * 2.0)))) +
                0.250000 * np.tanh(((data[:, 2]) + (((((
                        ((data[:, 14]) + ((((np.tanh((data[:, 10]))) + (data[:, 17])) / 2.0))) / 2.0)) + (
                                                          data[:, 21])) / 2.0)))) +
                0.250000 * np.tanh((((data[:, 9]) + ((((((((((
                        ((data[:, 9]) + ((((((data[:, 9]) / 2.0)) + (((data[:, 9]) * 2.0))) / 2.0))) / 2.0)) + (
                                                                (-1.0 * ((data[:, 21]))))) / 2.0)) * (data[:, 9]))) + ((
                        (((np.tanh((((data[:, 21]) / 2.0)))) / 2.0)) / 2.0))) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((((0.0)) + (data[:, 13])) / 2.0)) +
                0.250000 * np.tanh((((((((data[:, 2]) / 2.0)) * 2.0)) + (data[:, 7])) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 14]) * 2.0)) + (np.tanh((data[:, 14])))) / 2.0)) +
                0.250000 * np.tanh(np.tanh((((((((((-1.0 * ((((np.tanh((data[:, 9]))) / 2.0))))) / 2.0)) / 2.0)) + (
                    ((data[:, 3]) * (data[:, 13])))) / 2.0)))) +
                0.250000 * np.tanh((-1.0 * ((((((((
                        -1.0 * (((((((-1.0 * ((data[:, 18])))) - ((((0.0)) / 2.0)))) / 2.0))))) + (np.tanh(
                    ((((-1.0 * (((-1.0 * ((data[:, 22]))))))) / 2.0))))) / 2.0)) / 2.0))))))

    def GP_class_13(self, data):
        return (0.250000 * np.tanh(((((np.tanh((np.tanh((data[:, 2]))))) - ((10.0)))) - (np.tanh(
            (((((data[:, 7]) - ((((((6.0)) - ((4.87243366241455078)))) - ((9.0)))))) - ((14.80352973937988281)))))))) +
                0.250000 * np.tanh(((np.tanh((((((3.0)) + (data[:, 1])) / 2.0)))) - ((7.0)))) +
                0.250000 * np.tanh(
                    ((((np.tanh((np.tanh((np.tanh((data[:, 6]))))))) * 2.0)) - ((6.10337877273559570)))) +
                0.250000 * np.tanh(((((((data[:, 9]) + (data[:, 14]))) + (data[:, 14]))) + (data[:, 14]))) +
                0.250000 * np.tanh(((((((np.tanh(
                    ((((-1.0 * (((((((data[:, 2]) + ((10.0))) / 2.0)) / 2.0))))) - ((13.28130435943603516)))))) + (
                                            (10.0)))) - ((13.28130435943603516)))) + ((-1.0 * ((data[:, 5])))))) +
                0.250000 * np.tanh(((((((((((data[:, 20]) + (
                    ((((((((data[:, 20]) * 2.0)) * 2.0)) * 2.0)) + (data[:, 13])))) / 2.0)) + ((
                        (data[:, 9]) + (((data[:, 9]) + (((data[:, 6]) + (data[:, 9])))))))) / 2.0)) + (
                                          data[:, 20]))) + (data[:, 15]))) +
                0.250000 * np.tanh(((((((((data[:, 14]) / 2.0)) + ((((data[:, 9]) + (((data[:, 15]) + (((data[:, 5]) + (
                    (((((data[:, 9]) + (((data[:, 5]) + (data[:, 14]))))) + (
                        data[:, 9])) / 2.0))))))) / 2.0)))) * 2.0)) + (
                                        data[:, 14]))) +
                0.250000 * np.tanh((((data[:, 13]) + (data[:, 7])) / 2.0)) +
                0.250000 * np.tanh(
                    ((data[:, 9]) + ((((data[:, 3]) + ((((data[:, 21]) + (data[:, 3])) / 2.0))) / 2.0)))) +
                0.250000 * np.tanh((((data[:, 2]) + ((((((((((
                        (((((data[:, 20]) + (data[:, 1]))) + (((data[:, 2]) + (data[:, 14]))))) + (data[:, 2]))) + (
                                                                np.tanh((data[:, 2])))) / 2.0)) + (
                                                             ((data[:, 14]) + (data[:, 2]))))) + (
                                                           data[:, 14])) / 2.0))) / 2.0)))

    def GP_class_14(self, data):
        return (0.250000 * np.tanh((((((((((((5.54024744033813477)) - ((((((9.0)) * 2.0)) * 2.0)))) - (data[:, 0]))) - (
            (((7.0)) + ((4.74105215072631836)))))) - (((((7.0)) + (((data[:, 5]) * 2.0))) / 2.0)))) - ((9.0)))) +
                0.250000 * np.tanh(((((((((9.0)) - ((14.27298927307128906)))) - (data[:, 0]))) + (
                    np.tanh(((((12.77708148956298828)) - ((14.80560111999511719))))))) / 2.0)) +
                0.250000 * np.tanh(((((((0.0)) + ((-1.0 * ((((data[:, 22]) + (((((((
                        ((13.30077362060546875)) - (np.tanh((data[:, 4]))))) + ((
                        (data[:, 22]) / 2.0))) / 2.0)) / 2.0)))))))) / 2.0)) - (
                                        ((np.tanh((((data[:, 7]) - (data[:, 7]))))) / 2.0)))) +
                0.250000 * np.tanh(((((((data[:, 6]) + (data[:, 14])) / 2.0)) + (((np.tanh(((
                        (((data[:, 14]) + (data[:, 14]))) + (((data[:, 6]) - (((data[:, 5]) + ((
                        (np.tanh((data[:, 21]))) + (
                    (((data[:, 14]) + ((((data[:, 14]) + (data[:, 14])) / 2.0))) / 2.0)))))))))))) + (
                                                                                      data[:, 5])))) / 2.0)) +
                0.250000 * np.tanh((((((((((((((((((data[:, 2]) + (data[:, 16]))) + (
                    (((((data[:, 2]) / 2.0)) + (data[:, 9])) / 2.0))) / 2.0)) + (np.tanh((data[:, 10])))) / 2.0)) + (
                                               data[:, 9]))) + (data[:, 9])) / 2.0)) + (data[:, 21]))) + (
                                        data[:, 3]))) +
                0.250000 * np.tanh(
                    (((data[:, 13]) + (((((((((((data[:, 3]) / 2.0)) / 2.0)) / 2.0)) * (data[:, 3]))) / 2.0))) / 2.0)) +
                0.250000 * np.tanh(((data[:, 14]) + (((((((data[:, 14]) / 2.0)) + (data[:, 14]))) * 2.0)))) +
                0.250000 * np.tanh(data[:, 9]) +
                0.250000 * np.tanh(((data[:, 1]) / 2.0)) +
                0.250000 * np.tanh((((
                                         ((data[:, 2]) + (
                                             (((((data[:, 13]) + (data[:, 14]))) + (data[:, 13])) / 2.0)))) + (
                                         np.tanh((data[:, 2])))) / 2.0)))

    def GP_class_15(self, data):
        return (0.250000 * np.tanh((((((((4.0)) + ((((8.0)) / 2.0)))) - ((9.56193733215332031)))) - (
            ((((np.tanh(((13.93556308746337891)))) - (data[:, 18]))) + ((13.93556308746337891)))))) +
                0.250000 * np.tanh(((data[:, 4]) - ((((12.76094818115234375)) - (data[:, 8]))))) +
                0.250000 * np.tanh((((-1.0 * (((12.35446166992187500))))) - (((((12.35446166992187500)) + (
                    (((((12.35446166992187500)) - ((-1.0 * (((0.0))))))) - (data[:, 0])))) / 2.0)))) +
                0.250000 * np.tanh(((data[:, 21]) - ((14.46367263793945312)))) +
                0.250000 * np.tanh(((((data[:, 2]) + ((((data[:, 9]) + (
                    (-1.0 * (((-1.0 * (((-1.0 * ((((data[:, 7]) * (((data[:, 21]) * 2.0)))))))))))))) / 2.0)))) + (
                                        data[:, 9]))) +
                0.250000 * np.tanh(((((((((((data[:, 7]) + (data[:, 3])) / 2.0)) + (data[:, 20])) / 2.0)) - (
                    (((2.33354020118713379)) / 2.0)))) - ((((2.0)) + ((8.78930377960205078)))))) +
                0.250000 * np.tanh(((((((data[:, 14]) + (((((((data[:, 14]) + (((data[:, 15]) + (data[:, 14]))))) + (
                    ((data[:, 15]) + (data[:, 14]))))) + (((data[:, 14]) / 2.0)))))) + (data[:, 14]))) + (
                                        data[:, 14]))) +
                0.250000 * np.tanh(((((((((data[:, 13]) + (data[:, 13]))) + (((data[:, 13]) + (data[:, 12]))))) + (
                    data[:, 9]))) + ((((((data[:, 9]) + (
                    ((data[:, 13]) + ((((data[:, 9]) + (((((data[:, 13]) * 2.0)) / 2.0))) / 2.0))))) / 2.0)) * 2.0)))) +
                0.250000 * np.tanh((((data[:, 9]) + ((((((data[:, 9]) + (data[:, 9])) / 2.0)) / 2.0))) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 14]) + (data[:, 8]))) + (data[:, 14])) / 2.0)))

    def GP_class_16(self, data):
        return (0.250000 * np.tanh((((((
                (((((((9.37592792510986328)) - ((13.36316204071044922)))) + ((11.89122962951660156))) / 2.0)) - (
            (((9.0)) - (((((((data[:, 13]) / 2.0)) / 2.0)) + ((((-1.0 * ((((data[:, 4]) / 2.0))))) * 2.0)))))))) - (
                                          (12.24639320373535156)))) - ((11.89122581481933594)))) +
                0.250000 * np.tanh(((((np.tanh(((4.51821422576904297)))) - ((13.23712635040283203)))) - ((((
                        ((13.23712635040283203)) + (((data[:, 22]) - ((((11.40539550781250000)) - (((((9.0)) + (
                    (((((((data[:, 7]) + (((data[:, 17]) / 2.0))) / 2.0)) * 2.0)) / 2.0))) / 2.0)))))))) - (data[:,
                                                                                                            5]))))) +
                0.250000 * np.tanh(
                    ((data[:, 14]) - ((((4.0)) + (((((13.41770648956298828)) + ((12.42538261413574219))) / 2.0)))))) +
                0.250000 * np.tanh((((5.0)) - ((9.27778816223144531)))) +
                0.250000 * np.tanh(((((data[:, 16]) - (((((((data[:, 1]) - (((np.tanh((((((((9.29507255554199219)) - (
                    ((((((14.34461498260498047)) * 2.0)) + (((((7.0)) + ((9.10683441162109375))) / 2.0))) / 2.0)))) + ((
                    4.21145153045654297))) / 2.0)))) * 2.0)))) - ((9.10683441162109375)))) * 2.0)))) - (
                                        (((14.34461498260498047)) * 2.0)))) +
                0.250000 * np.tanh((((((((((data[:, 17]) / 2.0)) + (((data[:, 13]) + (data[:, 9])))) / 2.0)) + (
                    ((data[:, 17]) + (data[:, 9]))))) + (data[:, 0]))) +
                0.250000 * np.tanh(((data[:, 13]) + (((data[:, 14]) - (data[:, 21]))))) +
                0.250000 * np.tanh(
                    (((((data[:, 14]) + (data[:, 5])) / 2.0)) + (((((((data[:, 9]) / 2.0)) + (data[:, 14]))) / 2.0)))) +
                0.250000 * np.tanh((((((data[:, 17]) + (data[:, 9]))) + (((((data[:, 17]) / 2.0)) + (
                    (((((((((data[:, 9]) + (data[:, 9]))) - ((3.0)))) + (data[:, 7]))) + (
                        data[:, 9])) / 2.0))))) / 2.0)) +
                0.250000 * np.tanh(np.tanh(((((((((((((((data[:, 2]) + ((
                        (((((((((data[:, 22]) + (data[:, 21])) / 2.0)) + (data[:, 2])) / 2.0)) * 2.0)) + (
                    (((((data[:, 21]) * 2.0)) + (data[:, 21])) / 2.0)))))) + (data[:, 2])) / 2.0)) + (
                                                       data[:, 17])) / 2.0)) + (np.tanh((data[:, 8])))) / 2.0)) + (
                                                 data[:, 2]))))))

    def GP_class_17(self, data):
        return (0.250000 * np.tanh((((
                (data[:, 14]) - ((((data[:, 5]) + (((((14.43326377868652344)) + ((6.0))) / 2.0))) / 2.0)))) - (
                                        ((((((10.0)) + ((14.84462165832519531))) / 2.0)) - (
                                            (((7.0)) + (data[:, 13]))))))) +
                0.250000 * np.tanh((((((((((data[:, 14]) + (
                    ((((((((data[:, 11]) - ((9.71331787109375000)))) / 2.0)) + ((9.0)))) - ((3.0))))) / 2.0)) - (
                                            (10.0)))) - (
                                          (((10.0)) + ((((9.0)) - ((((((10.0)) - ((9.0)))) / 2.0)))))))) - (
                                        (10.0)))) +
                0.250000 * np.tanh((((((((((8.0)) - ((12.78277492523193359)))) - (data[:, 10]))) - (
                    np.tanh(((((((7.0)) - ((14.95321178436279297)))) - ((8.0)))))))) - ((14.95321178436279297)))) +
                0.250000 * np.tanh((((7.45682907104492188)) - ((14.98023796081542969)))) +
                0.250000 * np.tanh(((((((((((7.0)) + (
                    ((((np.tanh((((((((8.0)) * (data[:, 8]))) + (data[:, 11])) / 2.0)))) / 2.0)) - (
                        (8.0))))) / 2.0)) - (
                                            data[:, 20]))) - ((((8.0)) * 2.0)))) + ((8.0)))) +
                0.250000 * np.tanh(((data[:, 8]) + (
                    ((data[:, 4]) + (((np.tanh((data[:, 4]))) + (((data[:, 13]) + (data[:, 13]))))))))) +
                0.250000 * np.tanh(data[:, 14]) +
                0.250000 * np.tanh((((((
                        (((((data[:, 14]) + (np.tanh((((data[:, 3]) * (data[:, 13]))))))) + (data[:, 14]))) + (
                    data[:, 0]))) + (data[:, 8]))) / 2.0)) +
                0.250000 * np.tanh((((((data[:, 9]) + (np.tanh(((((((data[:, 19]) * (((((((data[:, 1]) + (
                    ((((((data[:, 21]) + (data[:, 9])) / 2.0)) + (data[:, 6])) / 2.0))) / 2.0)) + (
                                                                                           data[:, 17])) / 2.0)))) + (
                                                                      ((data[:,
                                                                        15]) / 2.0))) / 2.0))))) / 2.0)) * 2.0)) +
                0.250000 * np.tanh(((data[:, 1]) + (
                    ((((data[:, 3]) + (data[:, 15]))) + (
                        np.tanh(((((data[:, 20]) + (((data[:, 1]) * 2.0))) / 2.0)))))))))

    def GP_class_18(self, data):
        return (0.250000 * np.tanh((((
                ((((((12.59485149383544922)) - ((12.59485149383544922)))) - ((12.59485149383544922)))) - (
            (((11.47756862640380859)) / 2.0)))) - ((12.59485149383544922)))) +
                0.250000 * np.tanh((((((-1.0 * (((11.26121807098388672))))) - (((((
                        ((((11.33140659332275391)) - ((-1.0 * (((9.76293182373046875))))))) - (
                    ((((11.33140659332275391)) + ((8.0))) / 2.0)))) + ((9.76293182373046875))) / 2.0)))) - (
                                        data[:, 11]))) +
                0.250000 * np.tanh(((((((((((3.40250444412231445)) - (np.tanh(((14.86789989471435547)))))) - (
                    (14.86789989471435547)))) + ((
                        ((((data[:, 5]) + ((-1.0 * (((((14.86789989471435547)) * 2.0)))))) / 2.0)) - (
                    (14.86789989471435547))))) / 2.0)) - ((
                        (((14.86789989471435547)) + (((((7.0)) + ((-1.0 * ((data[:, 16]))))) / 2.0))) / 2.0)))) +
                0.250000 * np.tanh(((data[:, 18]) - ((11.84332561492919922)))) +
                0.250000 * np.tanh(((((np.tanh(((((8.96548557281494141)) * (data[:, 12]))))) - (
                    (-1.0 * ((((data[:, 2]) * (((((data[:, 4]) - ((-1.0 * ((data[:, 9])))))) - ((9.0))))))))))) - (
                                        (8.96548557281494141)))) +
                0.250000 * np.tanh(((((((10.0)) - (data[:, 12]))) + ((((((
                        (((data[:, 13]) - ((12.43707370758056641)))) - (
                    (((12.43707370758056641)) - (((data[:, 12]) / 2.0)))))) - ((
                        ((12.43707370758056641)) - ((((data[:, 13]) + ((12.43707370758056641))) / 2.0)))))) - (
                                                                          (12.43707370758056641))))) / 2.0)) +
                0.250000 * np.tanh((((
                        (((np.tanh((((np.tanh((data[:, 14]))) + (data[:, 13]))))) * 2.0)) + (data[:, 14]))) + (
                                        data[:, 13]))) +
                0.250000 * np.tanh(((np.tanh((data[:, 14]))) + (data[:, 14]))) +
                0.250000 * np.tanh(((((data[:, 9]) + (((((data[:, 13]) + (data[:, 4]))) + (data[:, 13]))))) / 2.0)) +
                0.250000 * np.tanh(((data[:, 15]) - ((((data[:, 3]) + ((5.0))) / 2.0)))))

    def GP_class_19(self, data):
        return (0.250000 * np.tanh((((((data[:, 13]) + (data[:, 13])) / 2.0)) + (
            (((((data[:, 8]) + ((((data[:, 14]) + (data[:, 14])) / 2.0))) / 2.0)) * 2.0)))) +
                0.250000 * np.tanh((((data[:, 3]) + (((data[:, 8]) + (np.tanh(((((((-1.0 * ((((((
                    9.05535793304443359)) + ((((data[:, 14]) + (((((((data[:, 1]) - (((data[:, 12]) * 2.0)))) * (
                    (-1.0 * ((data[:, 17])))))) / 2.0))) / 2.0))) / 2.0))))) + (data[:, 2]))) * 2.0))))))) / 2.0)) +
                0.250000 * np.tanh(((((((((data[:, 3]) + (((((data[:, 14]) + (data[:, 14]))) * 2.0)))) + (
                    (((data[:, 14]) + (np.tanh((((data[:, 14]) + (np.tanh((data[:, 14])))))))) / 2.0)))) * 2.0)) + (
                                        (((data[:, 14]) + (data[:, 2])) / 2.0)))) +
                0.250000 * np.tanh(((((data[:, 8]) + (((data[:, 4]) + (((data[:, 13]) + (data[:, 10]))))))) + ((
                        (((data[:, 13]) + (((((data[:, 13]) / 2.0)) + (data[:, 9]))))) + (
                    np.tanh(((((data[:, 8]) + (data[:, 8])) / 2.0)))))))) +
                0.250000 * np.tanh((((((data[:, 19]) + (
                    ((((data[:, 10]) + ((((data[:, 11]) + (data[:, 19])) / 2.0)))) + (data[:, 10])))) / 2.0)) + (
                                        data[:, 2]))) +
                0.250000 * np.tanh((((((data[:, 0]) + (((((
                        ((((data[:, 0]) + (((((data[:, 0]) - (data[:, 11]))) / 2.0))) / 2.0)) * 2.0)) + ((((((
                                                                                                                 np.tanh(
                                                                                                                     (
                                                                                                                         data[
                                                                                                                         :,
                                                                                                                         13]))) + (
                                                                                                                 ((data[
                                                                                                                   :,
                                                                                                                   0]) / 2.0))) / 2.0)) / 2.0))) / 2.0))) / 2.0)) * 2.0)) +
                0.250000 * np.tanh(((((-1.0 * ((data[:, 15])))) + (((data[:, 14]) + (data[:, 14])))) / 2.0)) +
                0.250000 * np.tanh((((
                        (((((data[:, 0]) + (data[:, 15])) / 2.0)) + (((data[:, 15]) * (data[:, 15])))) / 2.0)) - (
                                        data[:, 11]))) +
                0.250000 * np.tanh(data[:, 13]) +
                0.250000 * np.tanh(((((((((data[:, 18]) + ((((data[:, 4]) + (data[:, 18])) / 2.0)))) / 2.0)) + (
                    (((data[:, 18]) + (data[:, 4])) / 2.0)))) * (data[:, 4]))))


if __name__ == '__main__':

    # train = pd.read_csv('../data/train.csv')
    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
    train_basetable = preprocess(train)

    yards = train_basetable.pop('Yards')

    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1

    cat_features = []
    dense_features = []
    sss = {}
    lbls = {}
    X_train = train_basetable
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            # print(f)
            cat_features.append((col, len(train[col].unique())))
            lbl = LabelEncoder()
            X_train[col].fillna(-999, inplace=True)
            lbl.fit(list(X_train[col]) + [-999])
            X_train[col] = lbl.transform(list(X_train[col]))
            lbls[col] = lbl
        else:
            ss = StandardScaler()
            X_train[col].fillna(np.mean(X_train[col]), inplace=True)
            X_train.loc[:, col] = ss.fit_transform(X_train[col].values[:, None])
            dense_features.append(col)
            sss[col] = ss

    X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.15, random_state=0)

    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    """
    model = Sequential()
    model.add(Dense(512, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(199, activation='softmax'))
    """

    ## NEW
    ## Added: Batch normalization, Dropout

    model = Sequential()

    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(199, activation='softmax'))
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
    es = EarlyStopping(monitor='val_CRPS',
                       mode='min',
                       # restore_best_weights=True,
                       verbose=2,
                       patience=5)
    es.set_model(model)

    metric = Metric(model, [es], [(X_train, y_train), (X_val, y_val)])

    model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024)

    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    iter_test = env.iter_test()
    gp = GP()
    for (test_df, sample_prediction_df) in iter_test:
        X_test = preprocess(test_df)
        for col in X_test.columns:
            if X_test[col].dtype == 'object':
                # print(f)
                X_test[col] = lbls[col].transform(list(X_test[col]))
            else:
                X_test.loc[:, col] = sss[col].transform(X_test[col].values[:, None])

        y_pred_nn = model.predict(X_test)

        y_pred_gp = np.zeros((test_df.shape[0], 199))
        ans = gp.GrabPredictions(X_test)
        y_pred_gp[:, 96:96 + 20] = ans  # numpy才可以这样，pandas要用iloc

        y_pred = (.6 * y_pred_nn + .4 * y_pred_gp)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
        preds_df.iloc[:, :50] = 0
        preds_df.iloc[:, -50:] = 1
        env.predict(preds_df)

    env.write_submission_file()
