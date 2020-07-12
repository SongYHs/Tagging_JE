#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:37:17 2019

@author: Song Yunhua
"""
# import os
# os.environ["PATH"].append("/usr/local/cuda100/bin")
# import tensorflow.compat.v1 as tf #使用1.0版本的方法
# tf.disable_v2_behavior()
import numpy as np
import pickle as cPickle #python3
try:
    import pickle as cPickle    #python3
except:
    import cPickle              #python2
import keras.backend as K
import argparse, os
import PreparedataBert,process
from models import MModel
from Evaluate import evaluavtion_triple_1
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,Callback
from keras_bert import Tokenizer
learning_rate=1e-4
min_learning_rate=1e-5
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf
#
# config = tf.c
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)
# KTF.set_session(sess)
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
datadir='./data/'
config_path = datadir+"bert_dump/bert_config.json"
checkpoint_path = datadir+"bert_dump/bert_model.ckpt"
dict_path =datadir+"bert_dump/vocab.txt"

# config_path = "./data/bert_dump/bert_config.json"
# checkpoint_path ="./data/bert_dump/bert_model.ckpt"
# dict_path ="./data/bert_dump/vocab.txt"

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, default="./data/NYT_new/")
parser.add_argument('-output', type=str, default="./result/NYT_new/")
parser.add_argument('-e', type=str, default='./data/NYT_new/e2edata_Bert_k1.pkl')
parser.add_argument('-modelname', type=str, default='Zheng',choices=['Zheng', 'Ijcnn'])
parser.add_argument('-emb', type=str, default='word',choices=['bert', 'word'])
parser.add_argument('-embtrainable', type=int, default=1)#embtrainable
parser.add_argument('-a', type=int, default=10)
parser.add_argument('-b', type=int, default=10)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch', type=int, default=64)
parser.add_argument('-maxlen', type=int, default=50)
args = parser.parse_args()



token_dict = {}
with open(dict_path, 'r') as reader:#, python3 -> encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)
print(len(token_dict))





def test_model1(nn_model, testdata, index2word, sent_i2w, resultfile='', rfile='', kks=1):
    index2word[0] = ''
    xbatch = np.asarray(testdata[0], dtype="int32")

    sent_i2w[0] = 'UNK'
    senttext = []
    for sent in xbatch:
        stag = []
        for wi in range(len(sent)):
            token = sent_i2w[sent[-wi - 1]]
            stag.append(token)
        senttext.append(stag)
    ybatch = [np.asarray(testdata[1][i], dtype="int32") for i in range(len(testdata[0]))]  # [:50]

    testresult = []
    x=xbatch
    predictions = nn_model.predict(x)  # [:,:,1:-1]
    max_s=len(x[0])
    for si, sent in enumerate(predictions):
        # sent = predictions[si]
        ptag = []
        ptagi = ["O"]
        for ii, word in enumerate(sent):
            next_index = np.argmax(word)
            # if next_index != 0:
            next_token = index2word[next_index]
            ptagi.append(next_token)
            if not (ii+1) % (max_s-2):
                ptag.extend(ptagi+["O"])
                ptagi=["O"]
        #ptag.extend(ptagi + ["O"])
        senty = ybatch[si]
        ttag = []
        for word in senty:
            next_token = index2word[word]
            ttag.append(next_token)
        result = []
        result.append(ptag)
        result.append(ttag)
        testresult.append(result)
    cPickle.dump(testresult, open(resultfile, 'wb'))
    kk = len(ptag) // len(stag)

    P2, R2, F2, numpr2, nump2, numr2, numprSeq, numpSeq, numrSeq = evaluavtion_triple_1(testresult, max_s=max_s, kk=kk)
    print ('      New PRF    =', P2, R2, F2, numpr2, nump2, numr2)
    print('      kk=', str(kk), numprSeq, numpSeq, numrSeq)
    return P2, R2, F2, numpr2, nump2, numr2, numprSeq, numpSeq, numrSeq# P, R, F,P0,R0,F0,P1,R1,F1,




import time

""""""


def Loss(y_true, y_pred):
    loss0 = K.zeros_like(y_true[:, :1, 0])
    for i in range(kks):
        p1 = y_pred[:, i * max_tokens:i * max_tokens + max_tokens, :]
        w1 = K.sum(y_true[:, i * max_tokens:i * max_tokens + max_tokens, 2:], axis=-1)
        lossi = K.zeros_like(y_true[:, :max_tokens, 0])
        for j in range(kks):
            p2 = y_pred[:, j * max_tokens:j * max_tokens + max_tokens, :]
            l = K.mean(K.square(p1 - p2), axis=-1) * w1.any(axis=-1, keepdims=True)
            w2 = K.sum(y_true[:, j * max_tokens:j * max_tokens + max_tokens, 2:], axis=-1)
            lossi += l * w2.any(axis=-1, keepdims=True)
        loss0 = K.concatenate((loss0, lossi), axis=-1)
    #    return K.categorical_crossentropy(y_pred,y_true)-beta*loss0[:,1:]
    return K.categorical_crossentropy(y_true, y_pred) - beta * loss0[:, 1:]


""" """



def get_training_batch_xy_bias(inputsX, inputsY, max_s, max_t,
                          batchsize, vocabsize, target_idex_word,lossnum,shuffle=False,indices=None,Ki=1):
    assert len(inputsX) == len(inputsY)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(indices) - batchsize + 1, batchsize):#len(inputsX) - batchsize + 1, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            x = np.zeros((batchsize, max_s)).astype('int32')
            y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
            for idx, s in enumerate(excerpt):
                x[idx,] = inputsX[s]
                for idd in range(Ki):
                    if idd*max_s >= len(inputsY[s]):
                        break
                    for idx2, word in enumerate(inputsY[s][idd*max_s+1:idd*max_s+max_s-1]):
                        targetvec = np.zeros(vocabsize + 1)
                        wordstr=''
                        if word!=0:
                            wordstr = target_idex_word[word]
                        if wordstr.__contains__("E"):
                            targetvec[word] = lossnum
                        else:
                            targetvec[word] = 1
                        y[idx, idd*(max_s-2)+idx2,] = targetvec
            yield x,y











def generate_val(length):
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices[:int(length*0.9)],indices[int(length*0.9):]


def train_e2e_model(eelstmfile, modelfile, resultdir, npochos,emb='word',
                    lossnum=1, batch_size=256, retrain=False, max_t=1,loss="categorical_crossentropy"):
    class Evaluate(Callback):
        def __init__(self):
            self.F1 = []
            self.best = 0.
            self.passed = 0
            self.stage = 0

        def on_batch_begin(self, batch, logs=None):
            """第一个epoch用来warmup，第二个epoch把学习率降到最低
            """
            if self.passed < self.params['steps']:
                lr = (self.passed + 1.) / self.params['steps'] * learning_rate
                K.set_value(self.model.optimizer.lr, lr)
                self.passed += 1
            elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
                lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
                lr += min_learning_rate
                K.set_value(self.model.optimizer.lr, lr)
                self.passed += 1

        def on_epoch_end(self, epoch, logs=None):
            prf = test_model1(nn_model, testdata, target_idex_word, sourc_idex_word,
                              resultdir + 'result_val' + '.pkl',
                              kks=kks)
            precision, recall, f1 = prf[:3]
            self.F1.append(f1)
            if f1 > self.best:
                self.best = f1
                nn_model.save_weights(modelfile+'best_model.weights')
            print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    traindata, testdata, source_W,source_vob, sourc_idex_word, target_vob, target_idex_word, max_s \
        = cPickle.load(open(eelstmfile, 'rb'))

    # x_train = traindata[0]#[:5000]
    # y_train = traindata[1]#[:5000]
    for s in traindata[0]:
        s.reverse()
    for s in testdata[0]:
        s.reverse()

    hiddendim = 256
    print('len_s:', max_s, 'len_t:', max_t, 'hidden_dim:', hiddendim)
    print('start train')
    maxi = 0
    # kwag={}
    ind_train,ind_val=generate_val(len(traindata[0]))
    if emb=='bert':
        kwag={'config_path':config_path, 'checkpoint_path':checkpoint_path}
    else:
        kwag={'svb':len(source_W)+1,'input_seq_lenth':max_s,'source_W':source_W}
    # print(kwag)
    nn_model = MModel(args.modelname,hiddendim,len(target_vob), args.emb,emb_trainable=args.embtrainable,loss=loss,kw=kwag).model
    print(nn_model.summary())
    f = get_training_batch_xy_bias(traindata[0], traindata[1], max_s, max_t, \
                                   batch_size, len(target_vob), target_idex_word, lossnum, shuffle=True,
                                   indices=ind_train,Ki=kks)
    f_val = get_training_batch_xy_bias(traindata[0], traindata[1], max_s, max_t, \
                                       batch_size, len(target_vob), target_idex_word, lossnum, shuffle=True,
                                       indices=ind_val,Ki=kks)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4)
    EVA = Evaluate()
    h = nn_model.fit_generator(f, steps_per_epoch=len(ind_train) // batch_size, epochs=npochos, verbose=1, \
                               validation_data=f_val, validation_steps=len(ind_val) // batch_size,
                               callbacks=[early_stopping,  EVA])
#     print('  test')
#     prf = test_model1(nn_model, testdata, target_idex_word, sourc_idex_word,
#                       resultdir + 'result_test.pkl',
#                       kks=kks)  # ,rfile=resultdir+"json_result"+str(saveepoch)+'.json')
#     print('f1: %.4f, precision: %.4f, recall: %.4f\n' % (prf[2], prf[1], prf[0]))
#     print(h.history['loss'], maxi)
#     prf = []
    return h

def get_e2edatafile(emb,trainfile,testfile,labelfile,w2vfile):

    print("Precess data....")
    raw_train = trainfile
    seq_train = raw_train[:-5] + args.emb + '_all.json'
    raw_test = testfile
    seq_test = raw_test[:-5] + args.emb + '_all.json'
    if emb=='bert':
        tag_sent=PreparedataBert.tag_sent
        _ = tag_sent(raw_train, seq_train, labelfile, tokenizer)
        _ = tag_sent(raw_test, seq_test, labelfile, tokenizer)
        PreparedataBert.get_data_e2e(seq_train, seq_test, labelfile, token_dict, e2edatafile, maxlen=maxlen)
    else:
        tag_sent = process.tag_sent
        tag_sent(raw_train, seq_train, labelfile)
        tag_sent(raw_test, seq_test, labelfile)
        process.get_data_e2e(seq_train, seq_test, labelfile, w2vfile, e2edatafile, maxlen=50)

    print("Data Ok")




kks = 1 if args.modelname=='Zheng' else 2 ##################
max_tokens = 50

beta = 10
if __name__ == "__main__":
    e2edatafile = args.e
    alpha = args.a
    maxlen = args.maxlen
    batch = args.batch


    npoch = args.epoch
    max_s= args.maxlen
    max_t=kks * (max_s-2) if args.emb=="bert" else kks * max_s
    modeldir = args.output + 'model_' + args.modelname+args.emb + '/'
    resultdir = args.output + 'result_' + args.modelname+args.emb+ '/'
    print(modeldir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    if not os.path.exists(e2edatafile):
        trainfile = args.input + 'train.json'
        testfile = args.input + 'test.json'
        labelfile = args.input + 'label.txt'
        w2vfile = args.input + "w2v.pkl"
        get_e2edatafile(args.emb,trainfile,testfile,labelfile,w2vfile)
    loss = "categorical_crossentropy"#Loss if args.modelname=='Ijcnn' else
    train_e2e_model(e2edatafile, modeldir , resultdir,max_t=max_t,emb=args.emb,
                             npochos=npoch, lossnum=alpha, batch_size=batch,loss=loss)
