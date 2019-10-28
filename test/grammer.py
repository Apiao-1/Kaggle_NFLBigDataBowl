import numpy as np
import pandas as pd

# warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')[:200]
    print(train.head())
    # tmp = np.zeros((len(train), 6))
    # print(tmp.shape)
    tmp1 = train.iloc[:, :9] = 0
    # print(tmp1.shape)
    # train[:, 3:9] = tmp
    # train[:, -5:] = 5 * np.ones(len(train))
    print(train.head())

