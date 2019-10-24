###raw mae
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error
import sqlite3
import xgboost as xgb
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization

# from kaggle.competitions import nflrush
# env = nflrush.make_env()
if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')

    train.loc[train.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"

    train.loc[train.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"

    train.loc[train.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"

    train.loc[train.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"

    train['is_run'] = train.NflId == train.NflIdRusher
    train_single = train[train.is_run == True]


