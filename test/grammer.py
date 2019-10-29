import numpy as np
import pandas as pd

# warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

def drop(train):
    # drop_cols += ["Orientation", "Dir"]

    play_drop = ["GameId", 'PlayId', "TimeHandoff", "TimeSnap", "GameClock", "DefensePersonnel", "OffensePersonnel",
                 'FieldPosition', 'PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                 'HomeScoreBeforePlay','VisitorScoreBeforePlay','TeamOnOffense','Stadium']
    player_drop = ['DisplayName', 'PlayerBirthDate', "IsRusher", "NflId", "NflIdRusher", "Dir",
                   'Dir_rad', 'Ori_rad',"PlayDirection",'Orientation','Rusher_X','Rusher_Y',
                   'dist_to_rusher','time_to_rusher']
    environment_drop = ["WindSpeed", "WindDirection", "Season", "GameWeather",'Location','GameWeather_process'
                        'Turf']
    drop_cols = player_drop + play_drop + environment_drop
    train.drop(drop_cols, axis=1, inplace=True)
    return train

def preprocess(train):
    ## play是否发生在控球方所在的半场
    train['own_field'] = (train['FieldPosition'] == train['PossessionTeam']).astype(int)
    ## 主队持球或是客队持球
    train['process_type'] = (train['PossessionTeam'] == train['HomeTeamAbbr']).astype(int)

    ## PlayDirection
    train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')
    # 离自家球门的实际码线距离
    train['dist_to_end_train'] = train.apply(
        lambda x: (100 - x.loc['YardLine']) if x.loc['own_field'] == 1 else x.loc['YardLine'], axis=1)
    # ? https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    # train['dist_to_end_train'] = train.apply(lambda row: row['dist_to_end_train'] if row['PlayDirection'] else 100 - row['dist_to_end_train'],axis=1)
    # train.drop(train.index[(train['dist_to_end_train'] < train['Yards']) | (train['dist_to_end_train'] - 100 > train['Yards'])],inplace=True)

    # 统一进攻方向 https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
    train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi / 180.0
    train['X_std'] = train.X
    train.loc[~train.PlayDirection, 'X_std'] = 120 - train.loc[~train.PlayDirection, 'X']
    train['Y_std'] = train.Y
    train.loc[~train.PlayDirection, 'Y_std'] = 160 / 3 - train.loc[~train.PlayDirection, 'Y']
    train['Dir_std'] = train.Dir_rad
    train.loc[~train.PlayDirection, 'Dir_std'] = np.mod(np.pi + train.loc[~train.PlayDirection, 'Dir_rad'], 2 * np.pi)

    # 分方向的速度
    train["Dir_std_sin"] = train["Dir_std"].apply(lambda x: np.sin(x))
    train["Dir_std_cos"] = train["Dir_std"].apply(lambda x: np.cos(x))
    train['S_horizontal'] = train['S'] * train['Dir_std_cos']
    train['S_vertical'] = train['S'] * train['Dir_std_sin']

if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')[:200]
    train = preprocess(train)

