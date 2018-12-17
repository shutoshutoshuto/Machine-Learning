from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# データ読み込み
# kaiki_df = pd.read_csv("C:/Users/aizawa/Documents/WinPython-64bit-3.6.3.0Qt5/scripts/20180719/Switch_kakaku.csv", encoding="shift-jis")
kaiki_df = pd.read_csv("C:/Users/aizawa/Desktop/programing/Regression/optool.csv")
# print(kaiki_df)


# kaiki_df.drop(kaiki_df.columns[np.isnan(kaiki_df).any()], axis=1)
kaiki_df_except = kaiki_df.drop("baa", axis=1)
X = kaiki_df_except.as_matrix()

# 目的変数に "quality (品質スコア)" を利用
Y = kaiki_df["baa"].as_matrix()


# print(X)
# print(Y)



# 重回帰分析
# fit_intercept
# False に設定すると切片を求める計算を含めない。目的変数が原点を必ず通る性質のデータを扱うときに利用。 (デフォルト値: True)
# normalize
# True に設定すると、説明変数を事前に正規化します。 (デフォルト値: False)
# copy_X
# メモリ内でデータを複製してから実行するかどうか。 (デフォルト値: True)
# n_jobs
# 計算に使うジョブの数。-1 に設定すると、すべての CPU を使って計算します。 (デフォルト値: 1)
lmLR = linear_model.LinearRegression(fit_intercept=True,
                                             normalize=False,
                                             copy_X=True,
                                             n_jobs=1).fit(X, Y)


lmLR_variable = pd.DataFrame({"categoly":kaiki_df_except.columns,
                              "Coefficients":lmLR.coef_})
lmLR_intercept = pd.DataFrame({"categoly":["intercept"],
                               "Coefficients":lmLR.intercept_})

lmLR_function = lmLR_variable.append(lmLR_intercept)
print(lmLR_function)
