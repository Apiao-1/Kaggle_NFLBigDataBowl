# https://www.kaggle.com/bestpredict/location-eda-8eb410/comments
import os
TRAIN_ABLE_FALSE=True
if TRAIN_ABLE_FALSE:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda
from keras.models import Model
import re
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from keras.layers import BatchNormalization
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])

def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
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
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
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

def transform_time_all(str1, quarter):
    if quarter <= 4:
        return 15 * 60 - (int(str1[:2]) * 60 + int(str1[3:5])) + (quarter - 1) * 15 * 60
    if quarter == 5:
        return 10 * 60 - (int(str1[:2]) * 60 + int(str1[3:5])) + (quarter - 1) * 15 * 60

def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_Y(y_coordinate, play_direction):
        if play_direction == 'left':
            return 160.0/3 - y_coordinate
        else:
            return y_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
            # return yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)
            # return 100 - yardline

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    # https://www.kaggle.com/apiao1/model-nn/notebook?scriptVersionId=23555246
    # def new_orientation(angle, play_direction):
    #     if play_direction == 'left':
    #         new_angle = (270.0 - angle) % 360
    #         return new_angle
    #     else:
    #         return (90 - angle) % 360
    #
    # https: // www.kaggle.com / peterhurford / something - is -the - matter -with-orientation
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
        df['Y'] = df[['Y', 'PlayDirection']].apply(lambda x: new_Y(x[0], x[1]), axis=1) #v24
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
            .agg({'dist_to_back': ['min', 'max', 'mean', 'std'],
                  'X': ['mean', 'std'],
                  # 'Y': ['mean', 'std'],
                  # 'Y': ['max', 'min'],
                  }) \
            .reset_index()
        player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field',
                                   'back_moving_down_field',
                                   'min_dist', 'max_dist', 'mean_dist', 'std_dist',
                                   'X_mean','X_std',
                                   # 'Y_mean', 'Y_std',
                                   ]
        return player_distance

    def other(df, carriers):
        player_distance = df[['GameId', 'PlayId', 'NflId', 'X', 'Y', 'Position']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X', 'Y', 'back_X', 'back_Y']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        # Rusher距QB的距离，训练集23171 中有23290个QB，待确认缺失数据的处理
        QB_distance = player_distance[player_distance['Position'] == 'QB'][['dist_to_back', 'GameId', 'PlayId']].rename(
            columns={'dist_to_back': 'dist_QB'})
        player_distance = pd.merge(player_distance, QB_distance, on=['GameId', 'PlayId'], how='inner')
        print(player_distance.head())

        return player_distance[['GameId', 'PlayId', 'dist_QB']]

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY']

        defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][
            ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        defense['def_dist_to_back'] = defense[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        defense = defense.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back': ['min', 'mean', 'std'],
                  'X': ['mean', 'std'],
                  # 'Y': ['mean', 'std'],
                  }) \
            .reset_index()
        defense.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_mean_dist', 'def_std_dist',
                           'X_mean','X_std',
                           # 'Y_mean', 'Y_std',
                           ]

        return defense

    def offense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY']

        defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        defense = defense[defense['Team'] == defense['RusherTeam']][
            ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        defense['of_dist_to_back'] = defense[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        defense = defense.groupby(['GameId', 'PlayId']) \
            .agg({'of_dist_to_back': ['min', 'max', 'mean', 'std']}) \
            .reset_index()
        defense.columns = ['GameId', 'PlayId', 'of_min_dist', 'of_max_dist', 'of_mean_dist', 'of_std_dist']

        return defense

    def static_features(df):

        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
        add_new_feas.append('PlayerHeight_dense')

        # df['PlayerBMI'] = 703 * (df['PlayerWeight'] / (df['PlayerHeight_dense']) ** 2)
        # add_new_feas.append('PlayerBMI')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        # df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        # add_new_feas.append('TimeDelta')

        # df['time_end'] = df.apply(lambda x: transform_time_all(x.loc['GameClock'], x.loc['Quarter']), axis=1)
        # add_new_feas.append('time_end')

        df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60 * 60 * 24 * 365.25
        df['PlayerAge'] = df.apply(
            lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        # df['GameClock_sec'] = df['GameClock'].apply(strtoseconds)
        # add_new_feas.append('GameClock_sec')


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
        # df["Orientation_ob"] = df["Orientation"].apply(lambda x: orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x: orientation_to_cat(x)).astype("object")
        add_new_feas.append("Dir_ob")
        # add_new_feas.append("Orientation_ob")


        df["Orientation_sin"] = df["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))
        df['S_horizontal'] = df['S'] * df['Dir_cos']
        df['S_vertical'] = df['S'] * df['Dir_sin']
        add_new_feas.append("S_vertical")
        add_new_feas.append("S_horizontal")
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

        # df["diffScoreBeforePlay_binary_ob"] = (
        #             train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"])
        # add_new_feas.append("diffScoreBeforePlay_binary_ob")

        # play是否发生在控球方所在的半场
        # df['own_field'] = (df['FieldPosition'].fillna('') == df['PossessionTeam']).astype(int)
        # add_new_feas.append("own_field")

        # df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
        # df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"
        #
        # df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
        # df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"
        #
        # df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
        # df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"
        #
        # df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
        # df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
        # 主队持球或是客队持球
        # df['process_type'] = (df['PossessionTeam'] == df['HomeTeamAbbr']).astype(int)
        # add_new_feas.append("process_type")
        # 是否为防守方
        # df['TeamOnOffense'] = "home"
        # df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        # df['IsOnOffense'] = df.Team == df.TeamOnOffense
        # add_new_feas.append("IsOnOffense")
        # df['Team'] = df['Team'].apply(lambda x: x.strip() == 'home')
        # add_new_feas.append("Team")

        static_features = df[df['NflId'] == df['NflIdRusher']][
            add_new_feas + ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',
                            'YardLine', 'Quarter', 'Down', 'Distance', 'DefendersInTheBox']].drop_duplicates()
        #         static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))
        static_features.fillna(-999, inplace=True)
        #         for i in add_new_feas:
        #             static_features[i] = static_features[i].fillna(np.mean(static_features[i]))

        return static_features

    def combine_features(relative_to_back, defense, static, other=None, deploy=deploy):
        df = pd.merge(relative_to_back, defense, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, static, on=['GameId', 'PlayId'], how='inner')
        # df = pd.merge(df, other, on=['GameId', 'PlayId'], how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId', 'PlayId'], how='inner')

        return df

    def personnel_features(df):
        personnel = df[['GameId', 'PlayId', 'OffensePersonnel', 'DefensePersonnel']].drop_duplicates()
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
        personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
        personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
        personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
        personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
        personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
        personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
        personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
        personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

        # Let's create some features to specify if the OL is covered
        personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
        personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
        # Let's create a feature to specify if the defense is preventing the run
        # Let's just assume 7 or more DL and LB is run prevention
        personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

        personnel.drop(['OffensePersonnel', 'DefensePersonnel'], axis=1, inplace=True)

        return personnel

    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB', 'OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl, lb, db, other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB', 'LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR', 'DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb, rb, wr, te, ol)

        return counts

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    # other = other(df, back_feats)
    def_feats = defense_features(df)
    # of_feats = offense_features(df)
    static_feats = static_features(df)
    # personnel = personnel_features(df)
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
            # val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s


def get_model(x_tr, y_tr, x_val, y_val):
    inp = Input(shape=(x_tr.shape[1],))
    # x = Dense(2048, input_dim=X.shape[1], activation='relu')(inp)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(1024, input_dim=X.shape[1], activation='elu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='elu')(x)
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
                       restore_best_weights=True,
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
    model.load_weights("best_model.h5")

    y_pred = model.predict(x_val)
    y_true = y_val
    val_s = metric_crps(y_true, y_pred)
    # crps_round = np.round(val_s, 7)

    return model, val_s


def predict(x_te):
    model_num = len(models)
    for k, m in enumerate(models):
        if k == 0:
            y_pred = m.predict(x_te, batch_size=1024)
        else:
            y_pred += m.predict(x_te, batch_size=1024)

    y_pred = y_pred / model_num

    return y_pred

TRAIN_OFFLINE = False

if __name__ == '__main__':
    if TRAIN_OFFLINE:
        train = pd.read_csv('../data/train.csv', dtype={'WindSpeed': 'object'})[:2200]
    else:
        train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

    outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()

    train_basetable = create_features(train, False)

    X = train_basetable.copy()
    yards = X.Yards

    y = np.zeros((yards.shape[0], 199))
    for idx, target in enumerate(list(yards)):
        y[idx][99 + target] = 1

    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    losses = []
    models = []
    mean_crps_csv = []

    s_time = time.time()

    for k in range(5):
        kfold = KFold(5, random_state=42 + k, shuffle=True)
        crps_csv = []
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
            # print("-----------")
            # print("-----------")
            tr_x, tr_y = X[tr_inds], y[tr_inds]
            val_x, val_y = X[val_inds], y[val_inds]
            model, crps = get_model(tr_x, tr_y, val_x, val_y)
            models.append(model)
            # print("the %d fold crps is %f" % ((k_fold + 1), crps))
            crps_csv.append(crps)
        mean_crps_csv.append(np.mean(crps_csv))
        print("5 folder crps is %f" % np.mean(crps_csv))

    print("mean crps is %f" % np.mean(mean_crps_csv))

    if TRAIN_OFFLINE == False:
        from kaggle.competitions import nflrush

        env = nflrush.make_env()
        iter_test = env.iter_test()

        for (test_df, sample_prediction_df) in iter_test:
            basetable = create_features(test_df, deploy=True)
            basetable.drop(['GameId', 'PlayId'], axis=1, inplace=True)
            scaled_basetable = scaler.transform(basetable)

            y_pred = predict(scaled_basetable)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

            preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
            env.predict(preds_df)

        env.write_submission_file()
