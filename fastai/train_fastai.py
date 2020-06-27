
from fastai_custom.imports import *
from fastai_custom.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn.metrics import accuracy_score

from sklearn import metrics


import os
PATH=os.path.join(os.getcwd(),'data')

df_raw = pd.read_csv(f'{PATH}/training/dataset.csv', low_memory=False)

train_cats(df_raw)

df_valid = pd.read_csv(f'{PATH}/validation/dataset.csv', low_memory=False)

df_raw = pd.concat([df_raw,df_valid], axis=0, ignore_index=True)

df, y, nas = proc_df(df_raw, 'Cover_Type')

def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 9836  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape

def print_score(m):
    res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)

m = RandomForestClassifier(n_estimators=200, n_jobs=-1,oob_score=True,min_samples_leaf=1,max_features='sqrt')
m.fit(X_train, y_train)
print_score(m)

t=m.estimators_[0].tree_
dectree_max_depth(t)

feat_imp = m.feature_importances_

features = X_valid.columns.values

feat_imp_list = [(item[0],round(item[1]*100,2)) for item in list(zip(features, feat_imp))]

feat_imp_list.sort(key=lambda x: x[1], reverse=True)

print(feat_imp_list)
