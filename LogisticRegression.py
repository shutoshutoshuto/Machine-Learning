from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import svm

# =============================================================================
# 不倫データ
# =============================================================================
df = sm.datasets.fair.load_pandas().data


for i in range(len(df)):
    if df['affairs'][i] > 0:
        df['affairs'][i] =1
    else:
        df['affairs'][i] =0


# =============================================================================
# dummy変数に変換
# =============================================================================
occ_dummies = pd.get_dummies(df.occupation)
hus_occ_dummies = pd.get_dummies(df.occupation_husb)


# =============================================================================
# カラム付け
# =============================================================================
occ_dummies.columns = ['occ1' ,'occ2' ,'occ3' ,'occ4' ,'occ5' ,'occ6']
hus_occ_dummies.columns = ['hocc1' ,'hocc2' ,'hocc3' ,'hocc4' ,'hocc5' ,'hocc6']


# =============================================================================
# ダミーデータフレームの作成
# =============================================================================
df_drop = df.drop(['occupation','occupation_husb'], axis=1)
dummies = pd.concat([occ_dummies,hus_occ_dummies], axis=1)
df_dummies1 = pd.concat([df_drop ,dummies ], axis=1)


# =============================================================================
#　X,Yのデータ作成
# 「多重共線性」
# ある説明変数が、他の説明変数を1つまたは複数で表現できる場合、多重共線性があるという。
# 例えば今回は、occ1は、occ2〜occ6の値で一意に決まる関係にある。(occ2〜6に1が1個以上あれば、occ1=0、そうでなければocc1=1)
# この場合、逆行列が計算できなかったり、計算できても得られる結果の信頼性が低くなってしまう。
# =============================================================================
X = df_dummies1.drop(['affairs','occ1','hocc1'],axis=1)

Y = df.affairs
Y = np.ravel(Y)

# =============================================================================
# ロジスティック回帰
# =============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

log_model = LogisticRegression() 
log_model.fit(X_train, Y_train)
train_score = log_model.score(X, Y)
print("training score    ",train_score)

test_predict = log_model.predict(X_test)
test_score = accuracy_score(Y_test, test_predict)
print("test score        ",test_score)


coeff_df = pd.DataFrame([X.columns ,log_model.coef_[0]]).T
print("coeficient\n",coeff_df)
