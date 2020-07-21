import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from data import getFmat
data1 = pd.read_csv('data/data.csv')
data2 = pd.read_csv('data/data_non.csv')
data = pd.concat([data1, data2])
data = data[['E1type', 'E2type', 'Relation']]

# label encode
label_encoder = LabelEncoder()
data['E1type'] = label_encoder.fit_transform(data['E1type'].values)
data['E2type'] = label_encoder.fit_transform(data['E2type'].values)
data['Relation'] = label_encoder.fit_transform(data['Relation'].values)
# fmat = getFmat()
# np.save('fmat.npy', fmat)
fmat = np.load('fmat.npy')
# new_data = np.hstack((data.values[:, :2], fmat))
new_data = fmat
label = data.values[:, 2]
print(new_data.shape, label.shape)
X_train, X_test, y_train, y_test = train_test_split(new_data,
                                                    label,
                                                    test_size=0.3,
                                                    random_state=5)

#加载numpy的数组到DMatrix对象
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
#1.训练模型
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 0
param['num_class'] = 22

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist)

pred = bst.predict(xg_test)
precision = precision_score(y_test, pred, average='macro')
f1_score = f1_score(y_test, pred, average='macro')
print("Precision: %.2f%%\nF1_score: %.2f%%" % (precision * 100.0, f1_score * 100))

