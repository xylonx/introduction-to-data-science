import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

wine = pd.read_csv("./input/wine.csv")

# 0-1标准化
minmax_scale = preprocessing.MinMaxScaler().fit(
    wine[['Alcohol', 'Malic acid']])
np_minmax = minmax_scale.transform(wine[['Alcohol', 'Malic acid']])

# z-score标准化
std_scale = preprocessing.StandardScaler().fit(wine[['Alcohol', 'Malic acid']])
np_std = std_scale.transform(wine[['Alcohol', 'Malic acid']])

# 0-1标准化后，两个特征变量的最大值和最小值
print('Min-value after 0-1 scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(
    np_minmax[:, 0].min(), np_minmax[:, 1].min()))
print('\nMax-value after 0-1 scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(
    np_minmax[:, 0].max(), np_minmax[:, 1].max()))

print("\n----------------------------\n")

# z-score标准化后，两个特征变量的均值和方差
print('Mean after z-score standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(
    np_std[:, 0].mean(), np_std[:, 1].mean()))
print('\nStandard deviation after z-score standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(
    np_std[:, 0].std(), np_std[:, 1].std()))


# 画出散点图
df_minmax = wine.copy()
df_std = wine.copy()
df_minmax[['Alcohol', 'Malic acid']] = np_minmax
df_std[['Alcohol', 'Malic acid']] = np_std

wine_contrast = pd.concat([wine, df_minmax, df_std], axis=0)
plt.figure(figsize=(10, 10))
sns.pairplot(hue="Class", data=wine_contrast,
             x_vars="Alcohol", y_vars="Malic acid")
plt.show()
