import numpy as np
import pandas as pd


def get_cdf_df(yards_array):
    pdf, edges = np.histogram(yards_array, bins=199,range=(-99, 99), density=True) # density=True返回的是落在每个分区的概率
    cdf = pdf.cumsum().clip(0, 1)
    cdf_df = pd.DataFrame(data=cdf.reshape(-1, 1).T,
                          columns=['Yards' + str(i) for i in range(-99, 100)])
    return cdf_df


if __name__ == '__main__':
    train_plays = pd.read_csv('../data/train.csv', usecols=['PlayId', 'Yards']).drop_duplicates('PlayId') # (23171,)
    test_play_cdf = get_cdf_df(train_plays.Yards.values)
