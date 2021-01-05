from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# 加载相应的数据集
def load_data(data_path):
    data = pd.read_csv(data_path)
    label = data['label'][:350000]
    data = data.iloc[:350000, 1:]
    # 要把类别数据变成数字索引才能进行训练
    sparse_feats = []
    dense_feats = []
    for f in data.columns:
        if f[0] == 'C':
            sparse_feats.append(f)
            data[f].fillna('-1', inplace=True)
            vec = data[f].unique().tolist()
            le = LabelEncoder()
            le.fit(vec)
            data[f] = le.transform(data[f].values.tolist())
        else:
            dense_feats.append(f)
            data[f].fillna(0, inplace=True)

    # 这里设置一样random_state，目的是为了保持和下面的GBDT模型一致
    return data, label


# GBDT进行特征选择
def select_feature(x_train, y_train):
    gbm = GradientBoostingClassifier(n_estimators=50, random_state=10, max_depth=8)
    gbm.fit(x_train, y_train)
    train_new_feature = gbm.apply(x_train)
    print(train_new_feature.shape)
    train_new_feature = train_new_feature.reshape(-1, 50)
    enc = OneHotEncoder()
    enc.fit(train_new_feature)
    new_feature = np.array(enc.transform(train_new_feature).toarray())
    return new_feature


# 进行参数的训练
def train_step(new_features_data, train_label):
    lr_model = LogisticRegression(random_state=10, penalty='l2', max_iter=20)
    lr_model.fit(new_features_data, train_label)
    results = lr_model.predict(new_features_data)
    auc_result = roc_auc_score(results, train_label)
    return auc_result, lr_model


if __name__ == '__main__':
    total_data, total_label = load_data("./data/criteo_sampled_data.csv")
    # print(train_data.head(3))
    new_feature = select_feature(total_data, total_label)
    train_data, test_data, train_label, test_label = train_test_split(new_feature, total_label, test_size=0.3)
    acc_val, model = train_step(train_data, train_label)
    print("the acc_val on the training dataset is : {}".format(acc_val))

    # 下面对测试集进行操作
    # print(new_test_feature.shape)
    # print("\n")
    # print(new_feature.shape)
    test_result = model.predict(test_data)
    test_val = roc_auc_score(test_result, test_label)
    print("the acc_val on the test dataset is : {}".format(test_val))

