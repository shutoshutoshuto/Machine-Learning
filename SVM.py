from sklearn import datasets
from sklearn import svm
from sklearn import grid_search
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np



# =============================================================================
# データフレーム
# =============================================================================
df = datasets.load_digits()
X = df.data
Y = df.target


# =============================================================================
# クロスバリテーション
# =============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# =============================================================================
# SVM
# =============================================================================
clf = svm.SVC(kernel='linear', C=1, gamma=0.1)
clf.fit(X_train, Y_train)
train_score = clf.score(X_train, Y_train)
print("training score    ",train_score)

predict_clf = clf.predict(X_test)
test_score = accuracy_score(Y_test, predict_clf)
print("test score        ",test_score)
confusion_matrix = confusion_matrix(Y_test, predict_clf)
print("confusion_matrix\n",confusion_matrix)


# =============================================================================
# グリッドサーチ
# =============================================================================
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svr = svm.SVC(C=1)


clfg = grid_search.GridSearchCV(svr, parameters)
clfg.fit(X_train, Y_train)

print("\n+ ベストパラメータ:\n")
print(clfg.best_estimator_)

print("\n+ トレーニングデータでCVした時の平均スコア:\n")
for params, mean_score, all_scores in clfg.grid_scores_:
    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

print("\n+ テストデータでの識別結果:\n")
Y_true, Y_pred = Y_test, clfg.predict(X_test)
print(classification_report(Y_true, Y_pred))
