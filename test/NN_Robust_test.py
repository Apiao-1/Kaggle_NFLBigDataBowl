import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import lightgbm as lgb
import datetime
import warnings
import gc

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
        # blue_line
        new_yardline['YardLine'] = new_yardline[['PossessionTeam', 'FieldPosition', 'YardLine']].apply(
            lambda x: new_line(x[0], x[1], x[2]), axis=1)
        new_yardline['Yards_limit'] = 110 - new_yardline['YardLine']
        # green_line
        new_yardline['YardLine_next_down'] = new_yardline['YardLine'] + new_yardline['Distance']
        new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine', 'Yards_limit', 'YardLine_next_down']]

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
            ['GameId', 'PlayId', 'NflIdRusher', 'X', 'Y', 'Orientation', 'Dir', 'YardLine', 'YardLine_next_down']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_from_down'] = carriers['YardLine_next_down'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X': 'back_X',
                                            'Y': 'back_Y'})
        carriers = carriers[
            ['GameId', 'PlayId', 'NflIdRusher', 'back_X', 'back_Y', 'back_from_scrimmage'
                , 'back_oriented_down_field', 'back_moving_down_field','back_from_down'
             ]]

        return carriers


    def qb_features(df):
        qb = df[df['Position'] == 'QB'][['GameId', 'PlayId', 'X', 'Y']]
        qb.columns = ['GameId', 'PlayId','qbX', 'qbY']
        qb.drop_duplicates(['GameId', 'PlayId'],inplace=True)
        defense_qb = pd.merge(df, qb, on=['GameId', 'PlayId'], how='inner')

        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId','Team', 'X', 'Y']]
        carriers.columns = ['GameId', 'PlayId', 'RusherTeam', 'rushX', 'rushY']
        defense_qb = pd.merge(defense_qb, carriers, on=['GameId', 'PlayId'], how='inner')

        defense_qb['qbX'].fillna(defense_qb['rushX'])
        defense_qb['qbY'].fillna(defense_qb['rushY'])

        defense_qb = defense_qb[defense_qb['Team'] != defense_qb['RusherTeam']][['GameId', 'PlayId', 'X', 'Y', 'qbX', 'qbY','rushX', 'rushY']]
        defense_qb['dist_rusher_qb'] = defense_qb[['rushX', 'rushY', 'qbX', 'qbY']].apply(lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        # defense_qb['def_dist_to_qb'] = defense_qb[['X', 'Y', 'qbX', 'qbY']].apply(
        #     lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        defense_qb = defense_qb[['GameId', 'PlayId','dist_rusher_qb']]
        # defense_qb = defense_qb.groupby(['GameId', 'PlayId','dist_rusher_qb']) \
        #     .agg({'def_dist_to_qb': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt,'mad', np.ptp],
        #           }).reset_index()
        # defense_qb.columns = ['GameId', 'PlayId', 'dist_rusher_qb', 'def_min_qb', 'def_max_qb', 'def_mean_qb', 'def_std_qb',
        #                      'def_skew_qb', 'def_medn_qb', 'def_q80_qb', 'def_q30_qb', 'def_kurt_qb',
        #                      'def_mad_qb', 'def_ptp_qb',
        #                      ]
        return defense_qb

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId', 'PlayId', 'NflId', 'X', 'Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X', 'Y', 'back_X', 'back_Y']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

        player_distance = player_distance.groupby(
            ['GameId', 'PlayId', 'back_from_scrimmage',
             'back_oriented_down_field', 'back_moving_down_field','back_from_down'
             ]) \
            .agg({
            'dist_to_back': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad',np.ptp],
                  'X': ['mean', 'std'],
                  }) \
            .reset_index()

        player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage',
                                   'back_oriented_down_field', 'back_moving_down_field','back_from_down',
                                   'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'skew_dist', 'medn_dist',
                                   'q80_dist', 'q30_dist', 'kurt_dist', 'mad_dist', 'ptp_dist',
                                   'X_mean', 'X_std', ]

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A']]
        rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY', 'RusherS', 'RusherA']

        defense = pd.merge(df, rusher, on=['GameId', 'PlayId'], how='inner')
        defense_d = defense[defense['Team'] != defense['RusherTeam']][['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        defense_d['def_dist_to_back'] = defense_d[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        defense_d = defense_d.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt,'mad', np.ptp],
                  'X': ['min', 'max','mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'Y': ['min', 'max','mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  }).reset_index()
        defense_d.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist',
                             'def_skew_dist', 'def_medn_dist', 'def_q80_dist', 'def_q30_dist', 'def_kurt_dist',
                             'def_mad_dist', 'def_ptp_dist',
                             'def_X_min','def_X_max','def_X_mean', 'def_X_std', 'def_X_skew', 'def_X_median', 'def_X_q80', 'def_X_q30',
                             'def_X_kurt', 'def_X_mad', 'def_X_ptp',
                             'def_Y_min','def_Y_max','def_Y_mean', 'def_Y_std', 'def_Y_skew', 'def_Y_median', 'def_Y_q80', 'def_Y_q30',
                             'def_Y_kurt', 'def_Y_mad', 'def_Y_ptp',
                             ]

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
            .agg({'def_dist_to_back': ['max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt,'mad', np.ptp],
                  'X': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  'Y': ['min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp],
                  },
        ).reset_index()
        team_d.columns = ['GameId', 'PlayId',
                          'tm_max_dist', 'tm_mean_dist', 'tm_std_dist','tm_skew_dist', 'tm_medn_dist', 'tm_q80_dist', 'tm_q30_dist', 'tm_kurt_dist', 'tm_mad_dist','tm_ptp_dist',
                          'tm_X_min', 'tm_X_max','tm_X_mean', 'tm_X_std', 'tm_X_skew', 'tm_X_median', 'tm_X_q80', 'tm_X_q30','tm_X_kurt', 'tm_X_mad', 'tm_X_ptp',
                          'tm_Y_min', 'tm_Y_max','tm_Y_mean', 'tm_Y_std', 'tm_Y_skew', 'tm_Y_median', 'tm_Y_q80', 'tm_Y_q30','tm_Y_kurt', 'tm_Y_mad', 'tm_Y_ptp',
                        ]

        team_d_drop_rusher = team[(team['Team'] == team['RusherTeam']) & (team['NflId'] != team['NflIdRusher'])][['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
        team_d_drop_rusher['def_dist_to_back'] = team_d_drop_rusher[['X', 'Y', 'RusherX', 'RusherY']].apply(
            lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)
        team_d_drop_rusher = team_d_drop_rusher.groupby(['GameId', 'PlayId']) \
            .agg({'def_dist_to_back':['mean','min']}).reset_index()
        team_d_drop_rusher.columns = ['GameId', 'PlayId', 'tm_mean_drop_rusher','tm_min_drop_rusher']


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

        team = pd.merge(team_d, team_d_drop_rusher, on=['GameId', 'PlayId'], how='inner')
        team = pd.merge(team, team_s, on=['GameId', 'PlayId'], how='inner')

        return team

    # tested
    def static_features(df):
        df = df[df['NflId'] == df['NflIdRusher']]
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

        # df["Dir_ob"] = df["Dir"].apply(lambda x: orientation_to_cat(x))
        # add_new_feas.append("Dir_ob")

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

        def strtoseconds(txt):
            txt = txt.split(':')
            ans = int(txt[0]) * 60 + int(txt[1]) + int(txt[2]) / 60
            return ans

        df['GameClock_sec'] = df['GameClock'].apply(strtoseconds)
        df['score/seconds'] = (df['diffScoreBeforePlay'] / df['GameClock_sec'])
        df['score/seconds2'] = (df['diffScoreBeforePlay'] / (df['GameClock_sec'] + (4 - df['Quarter']) * 900))
        add_new_feas.append("score/seconds")
        add_new_feas.append("score/seconds2")

        static_features = df[
            add_new_feas + ['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',
                            'YardLine', 'Yards_limit','YardLine_next_down','Quarter', 'Down', 'Distance',
                            'DefendersInTheBox']]

        static_features.fillna(0, inplace=True)

        return static_features

    def combine_features(relative_to_back, defense, team, static, qb, deploy=deploy):
        df = pd.merge(relative_to_back, defense, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, team, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, static, on=['GameId', 'PlayId'], how='inner')
        df = pd.merge(df, qb, on=['GameId', 'PlayId'], how='inner')
        df = naodong_feature(df)


        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId', 'PlayId'], how='inner')

        return df

    def naodong_feature(train):
        # 两个球队和两条线的距离
        train['def_yardline'] = abs(train['def_X_mean'] - train['YardLine'])
        train['attack_yardline'] = abs(train['tm_mean_dist'] - train['YardLine'])
        train['def_yardline'] = abs(train['def_X_mean'] - train['YardLine_next_down'])
        train['attack_yardline'] = abs(train['tm_mean_dist'] - train['YardLine_next_down'])
        train['attack_def'] = abs(train['tm_mean_dist'] - train['def_X_mean'])

        train['drop_rusher2'] = abs(train['tm_mean_drop_rusher'] - train['YardLine_next_down'])
        train['drop_rusher3'] = abs(train['tm_mean_drop_rusher'] - train['YardLine'])
        train['drop_rusher1'] = abs(train['tm_mean_drop_rusher'] - train['X'])
        train['drop_rusher4'] = abs(train['def_X_mean'] - train['X'])

        # 平局剩下每个down要前进多少
        train['average_distance'] = (train['Distance'] / (5 - train['Down'])).values
        train['average_distance2'] = (train['Distance'] / (4 - train['Down'].replace(4, 3))).values

        train['s_gap'] = train['tm_mean_s'] - train['def_mean_s']
        train['a_gap'] = train['tm_mean_a'] - train['def_mean_a']
        train['as_gap'] = train['tm_mean_sa'] - train['def_mean_sa']
        return train


    # if deploy == False:
    #     df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[train['Season'] == 2017, 'Orientation'], 360)
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    qb_feats = qb_features(df)
    tm_feats = team_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, tm_feats, static_feats, qb_feats, deploy=deploy)

    # print(df.shape, back_feats.shape, rel_back.shape, def_feats.shape, static_feats.shape, basetable.shape)

    return basetable

# tested
def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) + np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    t_["nextS"] = t_["S"] + t_["A"]
    t_["nextSv"] = t_["nextS"] * np.cos(radian_angle)
    t_["nextSh"] = t_["nextS"] * np.sin(radian_angle)
    t_["Sv"] = t_["S"] * np.cos(radian_angle)
    t_["Sh"] = t_["S"] * np.sin(radian_angle)
    t_["Av"] = t_["A"] * np.cos(radian_angle)
    t_["Ah"] = t_["A"] * np.sin(radian_angle)
    t_["diff_ang"] = t_["Dir"] - t_["Orientation"]

    t_.fillna(0, inplace=True)
    t_.replace(np.nan, 0.0, inplace=True)
    t_.replace([np.inf, -np.inf], 0.0, inplace=True)

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
            val_s = metric_crps(y_valid, y_pred)
            # y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            # y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            # val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            # val_s = np.round(val_s, 8)
            logs['CRPS_score_val'] = val_s


def get_NN_model(x_tr, y_tr, x_val, y_val):
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
    crps = metric_crps(y_val, y_pred)
    # crps = np.round(crps, 8)
    # gc.collect()

    return model, crps


def get_LGBM_model(X_train, y_train, X_val, y_val):
    y_true = y_val
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        # 'n_estimators': 500,
        'learning_rate': 0.1,

        # 'num_leaves': 32,  # Original 50
        'max_depth': 5,

        'min_data_in_leaf': 49,  # min_child_samples
        # 'max_bin': 58,
        'min_child_weight': 19,

        "feature_fraction": 0.56,  # 0.9 colsample_bytree
        "bagging_freq": 9,
        "bagging_fraction": 0.9,  # 'subsample'
        "bagging_seed": 2019,

        # 'min_split_gain': 0.0,
        "lambda_l1": 0.21,
        "lambda_l2": 0.65,

        "boosting": "gbdt",
        'num_class': classify_type,  # 199 possible places
        # 'num_class': 199,  # 199 possible places
        'objective': 'multiclass',
        "metric": "None",
        # "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 2019,
    }
    num_round = 10000
    params['learning_rate'] = 0.02
    model = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=2000,
                      early_stopping_rounds=100, feval=crps_eval)
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    crps = metric_crps(y_true, y_pred)
    # gc.collect()
    return model, crps


def crps_eval(y_pred, dataset, is_higher_better=False):
    labels = dataset.get_label()
    y_true = np.zeros((len(labels), classify_type))
    for i, v in enumerate(labels):
        y_true[i, int(v):] = 1
    y_pred = y_pred.reshape(-1, classify_type, order='F')
    y_pred = np.clip(y_pred.cumsum(axis=1), 0, 1)
    return 'crps', np.mean((y_pred - y_true) ** 2), False


def metric_crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def predict(x_te):
    pred_NN = predict_single_modle(x_te, NN_models)
    pred_Lgbm = predict_single_modle(x_te, Lgbm_models)

    return weight_nn * pred_NN + weight_lgbm * pred_Lgbm


def predict_single_modle(x_te, model):
    model_num = len(model)
    for k, m in enumerate(model):
        if k == 0:
            y_pred = m.predict(x_te, batch_size=1024)
        else:
            y_pred += m.predict(x_te, batch_size=1024)
    y_pred = y_pred / model_num

    return y_pred


def weight_opt(oof_nn, oof_rf, y_true):
    weight_nn = np.inf
    best_crps = np.inf

    for i in np.arange(0, 1.01, 0.05):
        # gc.collect()

        crps_blend = np.zeros(oof_nn.shape[0])
        for k in range(oof_nn.shape[0]):
            crps_blend[k] = metric_crps(i * oof_nn[k, ...] + (1 - i) * oof_rf[k, ...], y_true)
        if np.mean(crps_blend) < best_crps:
            best_crps = np.mean(crps_blend)
            weight_nn = round(i, 2)

        print(str(round(i, 2)) + ' : mean crps (Blend) is ', round(np.mean(crps_blend), 6))

    print('-' * 36)
    print('Best weight for NN: ', weight_nn)
    print('Best weight for LGBM: ', round(1 - weight_nn, 2))
    #print('Best weight for RF: ', round(1-weight_nn, 2)
    print('Best mean crps (Blend): ', round(best_crps, 6))

    return weight_nn, round(1 - weight_nn, 2)

def sample_indices(df, fraction = 0.05):
    return df.sample(frac = fraction).index

def pseudo_random_changes(df):
    df.loc[sample_indices(df),'Dir'] = -9999 #negative
    df.loc[sample_indices(df),'Dir'] = 9999 #larger than 360
    df.loc[sample_indices(df),'Orientation'] = -9999 #negative
    df.loc[sample_indices(df),'Orientation'] = 9999 #larger than 360
    df.loc[sample_indices(df),'X'] = 200 #bigger than the field
    df.loc[sample_indices(df),'Y'] = 200
    df.loc[sample_indices(df),'X'] = -200 #negative
    df.loc[sample_indices(df),'Y'] = -200
    df.loc[sample_indices(df),'Position'] = 'QB' #multiple QB's
    df.loc[sample_indices(df),'PlayId'] = 20181202093044 #create duplicate playid
    df.loc[sample_indices(df),'OffenseFormation'] = 'NEW_VALUE'
    df.loc[sample_indices(df),'OffensePersonnel'] = 'NEW_VALUE'
    df.loc[sample_indices(df),'DefensePersonnel'] = 'NEW_VALUE'
    df.loc[sample_indices(df),'NflIdRusher'] = '999999'
    return df

def drop_rows_randomly(df, fraction = 0.05):
    np.random.seed(0)
    remove_n = round(fraction * df.shape[0])
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df = df.drop(drop_indices)
    return df

def insert_nan_randomly(df, fraction = 0.05):
    np.random.seed(0)
    return df.mask(np.random.random(df.shape) < fraction)


TRAIN_OFFLINE = False
CLASSIFY_NEGITAVE = -14  # must < 0
CLASSIFY_POSTIVE = 99  # 99， 75，53， 36
classify_type = CLASSIFY_POSTIVE - CLASSIFY_NEGITAVE + 1

if __name__ == '__main__':
    if TRAIN_OFFLINE:
        train = pd.read_csv('../input/train.csv', dtype={'WindSpeed': 'object'})
    else:
        train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
        # train.loc[train['Season'] == 2017, 'S'] = (train['S'][train['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570

    df = train

    df = (df.pipe(insert_nan_randomly)
          .pipe(drop_rows_randomly)
          .pipe(pseudo_random_changes)
          )

    train = df

    outcomes = train[['GameId', 'PlayId', 'Yards']].drop_duplicates()

    train_basetable = create_features(train, False)
    train_basetable = process_two(train_basetable)

    train = train_basetable.loc[
        (train_basetable['Yards'] >= CLASSIFY_NEGITAVE) & (train_basetable['Yards'] <= CLASSIFY_POSTIVE)]

    print("before delete:", train_basetable.shape)
    print("After delete:", train.shape)
    # print(train.head())
    X = train
    yards = X.Yards

    # y = np.zeros((yards.shape[0], 199))
    y = np.zeros((yards.shape[0], classify_type))
    for idx, target in enumerate(list(yards)):
        # y[idx][99 + target] = 1
        y[idx][-CLASSIFY_NEGITAVE + target] = 1

    X.drop(['GameId', 'PlayId', 'Yards'], axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    losses = []
    NN_models = []
    Lgbm_models = []
    NN_mean_crps_csv = []
    LGBM_mean_crps_csv = []

    loop = 2
    NN_pred = np.zeros((loop, len(y), len(y[0])))
    Lgbm_pred = np.zeros((loop, len(y), len(y[0])))
    for k in range(loop):
        kfold = KFold(6, random_state=2019 + 17 * k, shuffle=True)
        j = 0
        NN_crps_csv = []
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
            # j += 1
            # if j > 3:
            #     break
            tr_x, tr_y = X[tr_inds], y[tr_inds]
            val_x, val_y = X[val_inds], y[val_inds]

            # train NN model
            NN_model, NN_crps = get_NN_model(tr_x, tr_y, val_x, val_y)
            NN_models.append(NN_model)
            NN_crps_csv.append(NN_crps)
            NN_pred[k, val_inds, :] = NN_model.predict(val_x)

        NN_mean_crps_csv.append(np.mean(NN_crps_csv))
        print("9 folder NN crps is %f" % np.mean(NN_crps_csv))

    for k in range(1):
        kfold = KFold(5, random_state=2019 + 17 * k, shuffle=True)
        Lgbm_crps_csv = []
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
            tr_x, tr_y = X[tr_inds], y[tr_inds]
            val_x, val_y = X[val_inds], y[val_inds]

            Lgbm_model, Lgbm_crps = get_LGBM_model(tr_x, tr_y, val_x, val_y)
            Lgbm_models.append(Lgbm_model)
            Lgbm_crps_csv.append(Lgbm_crps)
            Lgbm_pred[k, val_inds, :] = Lgbm_model.predict(val_x, num_iteration=Lgbm_model.best_iteration)

        LGBM_mean_crps_csv.append(np.mean(Lgbm_crps_csv))
        print("9 folder LGBM crps is %f" % np.mean(Lgbm_crps_csv))

    Lgbm_pred[1, :, :] = Lgbm_pred[0, :, :]
    print("total mean NN crps is %f" % np.mean(NN_mean_crps_csv))
    print("total mean LGBM crps is %f" % np.mean(LGBM_mean_crps_csv))

    # get blend weight
    weight_nn, weight_lgbm = weight_opt(NN_pred, Lgbm_pred, y)

    if TRAIN_OFFLINE == False:
        from kaggle.competitions import nflrush

        env = nflrush.make_env()
        iter_test = env.iter_test()

        for (test_df, sample_prediction_df) in iter_test:
            basetable = create_features(test_df, deploy=True)
            basetable = process_two(basetable)
            Yards_limit = basetable['Yards_limit'][0]

            basetable.drop(['GameId', 'PlayId'], axis=1, inplace=True)
            scaled_basetable = scaler.transform(basetable)

            y_pred = predict(scaled_basetable)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)

            y_0 = np.zeros((len(y_pred), 99 + CLASSIFY_NEGITAVE))
            y_pred = np.concatenate((y_0, y_pred), axis=1)
            y_1 = np.ones((len(y_pred), 99 - CLASSIFY_POSTIVE))
            y_pred = np.concatenate((y_pred, y_1), axis=1)

            y_pred[:, (99 + int(Yards_limit)):] = 1

            preds_df = pd.DataFrame(data=y_pred, columns=sample_prediction_df.columns)
            env.predict(preds_df)

        env.write_submission_file()
