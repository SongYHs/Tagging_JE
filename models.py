#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:50:07 2019

@author: Song Yunhua
"""

from keras.layers import Input, Concatenate,Layer,Embedding,Activation,Dense,Dropout

from keras.layers import GRU,LSTM,RNN,Bidirectional,Reshape,Permute, Lambda,Add
from keras.models import Model
from keras import backend as K
from layers import LSTM_Decoder,MYGRUDecoder,GDCell,emb_init,top_k,DecodeCell,reshapecell
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_model, compile_model
import tensorflow as tf


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score # 两部分得分之和
    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs
    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)

class Attention(Layer):
    def __init__(self,cell,**kwargs):
        self.cell=cell
        self.supports_masking = True
        super(Attention,self).__init__(**kwargs)
    def build(self,input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.w=self.add_weight(name='W_att',
                               shape=(in_dim,out_dim),
                               initializer='uniform',
                               trainable=True)
        super(Attention,self).build(input_shape)
    def call(self,inputs):
        q,v,k=inputs
#        masks=self._collect_previous_mask(inputs)
        wq=K.dot(q,self.w)
        a = K.batch_dot(wq, k,[2,2])
        a = K.softmax(a)
        o=K.batch_dot(a,v,[2,1])
        return o
        
    def compute_output_shape(self,input_shape):
         return (input_shape[0][0],input_shape[0][1],
                input_shape[1][-1])
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        return mask[0]

# 1）郑孙聪那个+固定word2vec输入，跑一下kbp原始数据集，
# 2）郑孙聪那个+固定的bert向量，跑一下NYT，kbp以及webnlg的数据集，nyt和kbp都是原始的，webnlg用你处理过的就好，
# 3）你的ijcnn（词向量是word2vec）在webnlg上的实验，
# 4）你的ijcnn（词向量是bert）在nyt，kbp，以及webnlg上的实验（其中，加一组在nyt，kbp新数据集上实验）

class MModel():
    def __init__(self,modelname,hidden_dim, targetvocabsize, emb='bert', loss='categorical_crossentropy', optimizer='Adam',emb_trainable=True,kw={}):

        self.hidden_dim = hidden_dim
        self.loss = loss
        self.optimizer = optimizer
        self.emb_trainable= emb_trainable
        self.emb=emb #bert,vec
        self.targetvocabsize=targetvocabsize
        self.kw=kw
        if modelname == "Zheng":
            self.model = self.model_Zheng()
        elif modelname == "Ijcnn":
            self.model = self.model_Ijcnn()

    def model_Zheng(self,encode=GRU):
        kwag=self.kw
        # print(kwag)
        inputx = Input(shape=(None,))  # shape=(input_seq_lenth,))
        if self.emb=='bert':
            bert_model = load_trained_model_from_checkpoint(kwag['config_path'], kwag['checkpoint_path'])
            tp1 = Lambda(lambda x: K.zeros_like(x))(inputx)
            x1 = bert_model([inputx, tp1])
            embedding = Lambda(lambda x0: x0[:, 1:-1])(x1)
            for l in bert_model.layers:
                l.trainable = self.emb_trainable
        elif self.emb == 'word':
            embedding=Embedding(input_dim=kwag['svb']+1,output_dim=300, mask_zero=True,\
                                input_length=kwag['input_seq_lenth'],
                                embeddings_initializer=emb_init(kwag['source_W']),\
                                trainable=self.emb_trainable)(inputx)
            embedding = Dropout(0.3)(embedding)
        encoder = Bidirectional(encode(self.hidden_dim, return_sequences=True))(embedding)
        # encoder = Attention('gru')([encoder]*3)
        decoder = LSTM(self.hidden_dim, return_sequences=True)(encoder)
        p = Dense(self.targetvocabsize + 1)(decoder)
        p = Activation('softmax')(p)
        model = Model(inputs=inputx, outputs=p)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def model_Ijcnn(self, encode=GRU):
        kwag = self.kw
        inputx = Input(shape=(None,))  # shape=(input_seq_lenth,))
        if self.emb == 'bert':
            bert_model = load_trained_model_from_checkpoint(kwag['config_path'], kwag['checkpoint_path'])
            tp1 = Lambda(lambda x: K.zeros_like(x))(inputx)
            x1 = bert_model([inputx, tp1])
            embedding = Lambda(lambda x0: x0[:, 1:-1])(x1)
            for l in bert_model.layers:
                l.trainable = self.emb_trainable
        elif self.emb == 'word':
            embedding = Embedding(input_dim=kwag['svb']+1, output_dim=300, mask_zero=True, \
                                  input_length=kwag['input_seq_lenth'],
                                  embeddings_initializer=emb_init(kwag['source_W']), \
                                  trainable=self.emb_trainable)(inputx)
            embedding = Dropout(0.3)(embedding)

        encoder = Bidirectional(LSTM(self.hidden_dim, return_sequences=True))(embedding)
        # encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True))(encoder)
        decoder0 = LSTM(self.hidden_dim, return_sequences=True)(encoder)
        p0 = Dense(self.targetvocabsize + 1)(decoder0)
        p0 = Activation('softmax')(p0)

        decoder1 = LSTM(self.hidden_dim, return_sequences=True)(encoder)
        p1 = Dense(self.targetvocabsize + 1)(decoder1)
        p1 = Activation('softmax')(p1)

        p = Add()([p0, p1])
        model = Model(inputx, p)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'], loss_weights=[1.0])  # loss_weights)
        return model

