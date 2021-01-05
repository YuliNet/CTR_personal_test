import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


# 数据预处理
def pretrain_data(data_path):
    train_data = pd.read_csv(data_path)
    # 定义特征组
    dense_feats = [index for index, row in train_data.iteritems() if index[0] == 'I']
    sparse_feats = [index for index, row in train_data.iteritems() if index[0] == 'C']
    # 对于数值型特征，利用0.0填充未定义的值，然后通过log进行变换
    dense = train_data.copy()
    dense = dense[dense_feats].fillna(0.0)
    for f in dense_feats:
        dense[f] = dense[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    sparse = train_data.copy()
    sparse = sparse[sparse_feats].fillna('-1')
    for f in sparse_feats:
        sparse[f] = LabelEncoder().fit_transform(sparse[f])
    total_data = pd.concat([dense, sparse], axis=1)
    total_data['label'] = train_data['label']
    return dense, sparse, total_data, dense_feats, sparse_feats


class DeepFM:
    def __init__(self, sparse_feats, dense_feats, total_data, k=8):
        self.sparse_feats = sparse_feats
        self.dense_feats = dense_feats
        self.total_data = total_data
        self.k = k

    def create_model(self):
        dense_inputs = []
        dense_feats = self.dense_feats
        for f in dense_feats:
            _input = Input([1], name=f)
            dense_inputs.append(_input)
        concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
        fst_order_dense_layer = Dense(1)(concat_dense_inputs)

        # 单独对每个sparse的特征构造输入
        data = self.total_data
        sparse_inputs = []
        sparse_feats = self.sparse_feats
        for f in sparse_feats:
            _input = Input([1], name=f)
            sparse_inputs.append(_input)
        sparse_ld_embs = []
        for i, _input in enumerate(sparse_inputs):
            f = sparse_feats[i]
            voc_size = data[f].nunique()
            # 使用l2正则化 防止过拟合
            reg = tf.keras.regularizers.l2(0.5)
            _emb = Embedding(voc_size, 1, embeddings_regularizer=reg)(_input)
            print(_emb)
            _emb = Flatten()(_emb)
            sparse_ld_embs.append(_emb)
        first_order_sparse_layer = Add()(sparse_ld_embs)
        linear_part = Add()([fst_order_dense_layer, first_order_sparse_layer])
        sparse_kd_embs = []
        for i, _input in enumerate(sparse_inputs):
            f = sparse_feats[i]
            voc_size = data[f].nunique()
            reg = tf.keras.regularizers.l2(0.7)
            _emb = Embedding(voc_size, self.k, embeddings_regularizer=reg)(_input)
            sparse_kd_embs.append(_emb)
        concat_sparse_kd_layer = Concatenate(axis=1)(sparse_kd_embs)
        sum_kd_emb = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_layer)
        square_sum_kd_embed = Multiply()([sum_kd_emb, sum_kd_emb])
        square_kd_embed = Multiply()([concat_sparse_kd_layer, concat_sparse_kd_layer])
        sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)
        sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
        sub = Lambda(lambda x: x * 0.5)(sub)
        snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)

        # deep部分
        flatten_sparse_embed = Flatten()(concat_sparse_kd_layer)  # ?, n*k
        fc_layer = Dropout(0.5)(Dense(256, activation='relu')(flatten_sparse_embed))  # ?, 256
        fc_layer = Dropout(0.3)(Dense(256, activation='relu')(fc_layer))  # ?, 256
        fc_layer = Dropout(0.1)(Dense(256, activation='relu')(fc_layer))  # ?, 256
        fc_layer_output = Dense(1)(fc_layer)  # ?, 1

        # 将两个部分组合起来
        output_layer = Add()([linear_part, snd_order_sparse_layer, fc_layer_output])
        output_layer = Activation("sigmoid")(output_layer)

        model = Model(dense_inputs + sparse_inputs, output_layer)
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["binary_crossentropy", tf.keras.metrics.AUC()])

        # 模型的训练
        train_data = data.loc[:450000 - 1]
        valid_data = data.loc[450000:]

        train_dense_x = [train_data[f].values for f in dense_feats]
        train_sparse_x = [train_data[f].values for f in sparse_feats]
        train_label = [train_data['label'].values]

        val_dense_x = [valid_data[f].values for f in dense_feats]
        val_sparse_x = [valid_data[f].values for f in sparse_feats]
        val_label = [valid_data['label'].values]

        model.fit(train_dense_x + train_sparse_x,
                  train_label, epochs=20, batch_size=128,
                  validation_data=(val_dense_x + val_sparse_x, val_label),
                  )


if __name__ == '__main__':
    dense, sparse, total_data, dense_feats, sparse_feats = pretrain_data('./data/criteo_sampled_data.csv')
    deepfm = DeepFM(sparse_feats, dense_feats, total_data)
    deepfm.create_model()

