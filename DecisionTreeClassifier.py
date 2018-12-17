# =============================================================================
# http://scikit-learn.org/stable/modules/tree.html
# https://qiita.com/takahashi_yukou/items/5251bada1c3dc453c508
# https://pythondatascience.plavox.info/scikit-learn/scikit-learn%e3%81%a7%e6%b1%ba%e5%ae%9a%e6%9c%a8%e5%88%86%e6%9e%90
# =============================================================================


from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

df1 = load_iris()
X = df1.data
Y = df1.target

clf = tree.DecisionTreeClassifier(criterion='gini',
                                  splitter='best', 
                                  max_depth=3, 
                                  min_samples_split=2,
                                  min_samples_leaf=1, 
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None, 
                                  random_state=None,
                                  max_leaf_nodes=None, 
                                  class_weight=None, 
                                  presort=False)

clf.fit(X, Y)
predicted = clf.predict(X)

score = sum(predicted == Y) / len(Y)
print(score)


tree_data= tree.export_graphviz(clf, 
                               out_file=None, 
                               feature_names=df1.feature_names, 
                               class_names=df1.target_names,
                               filled=True, 
                               rounded=True)

graph = graphviz.Source(tree_data)



