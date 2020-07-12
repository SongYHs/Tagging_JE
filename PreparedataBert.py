import json
import numpy as np
try:
    import pickle as cPickle    #python3
except:
    import cPickle              #python2
from keras_bert import Tokenizer,get_base_dict



def find_index(sentTokens, emTokens ):
    index = 0
    stend = []
    while index+len(emTokens) < len(sentTokens):
        flag = 1
        for j in range(len(emTokens)):
            if sentTokens[index+j] != emTokens[j]:
                flag = 0
                break
        if flag:
            stend.append([index, index+len(emTokens)])
            index += len(emTokens)
        else:
            index += 1
    return stend




def tag_sent(source_json, tag_json, labeltxt, tokenizer):
    f=open(source_json,"r")
    fread=f.readlines()
    f.close()
    train_json_file=open(tag_json, "w")
    f = open(labeltxt, 'r')
    fr = f.readlines()
    Alabel = [lines.strip('\n') for lines in fr]
    f.close()
    ii=0
    c1,c2,c3=0,0,0
    for line in fread:
        kk = 1
        ii += 1
        flag = 0
        sent = json.loads(line.strip('\r\n'))
        sentTokens=tokenizer.tokenize(sent["sentText"])
        tags=["O"]*len(sentTokens)
        emIndexByTokens={}
        for em in sent["entityMentions"]:
            emTokens = tokenizer.tokenize(em["text"])[1:-1]
            emt="_".join(emTokens)
            if emt in emIndexByTokens:
                break
            start_end=find_index(sentTokens,emTokens)
            emIndexByTokens[emt]=start_end
        for rm in sent["relationMentions"]:
            if not rm["label"] == "None" and rm["label"] in Alabel:
                label = rm["label"]
                em1 = "_".join(tokenizer.tokenize(rm["em1Text"])[1:-1])
                em2 = "_".join(tokenizer.tokenize(rm["em2Text"])[1:-1])

                if em1 in emIndexByTokens and em2 in emIndexByTokens:
                    c1+=1
                    ind1 = emIndexByTokens[em1]
                    ind2 = emIndexByTokens[em2]
                    minind = len(sentTokens)
                    labelindex = []
                    for i1ind, i1 in enumerate(ind1):
                        for i2ind, i2 in enumerate(ind2):
                            if (i2[0] - i1[1]) * (i2[1] - i1[0]) > 0:
                                if minind > abs(i2[1] - i1[1]):
                                    minind = abs(i2[1] - i1[1])
                                    labelindex = [i1ind, i2ind]
                    if labelindex:
                        c2+=1
                        i1ind = labelindex[0]
                        i2ind = labelindex[1]
                        start1 = ind1[i1ind][0]
                        end1 = ind1[i1ind][1]
                        start2 = ind2[i2ind][0]
                        end2 = ind2[i2ind][1]
                        tag1Previous = []
                        tag2Previous = []
                        if end1 - start1 == 1:
                            tag1Previous.append(label + "__E1S")
                        elif end1 - start1 == 2:
                            tag1Previous.append(label + "__E1B")
                            tag1Previous.append(label + "__E1L")
                        else:
                            tag1Previous.append(label + "__E1B")
                            for ei in range(start1 + 1, end1 - 1):
                                tag1Previous.append(label + "__E1I")
                            tag1Previous.append(label + "__E1L")
                        if end2 - start2 == 1:
                            tag2Previous.append(label + "__E2S")
                        elif end2 - start2 == 2:
                            tag2Previous.append(label + "__E2B")
                            tag2Previous.append(label + "__E2L")
                        else:
                            tag2Previous.append(label + "__E2B")
                            for ei in range(start2 + 1, end2 - 1):
                                tag2Previous.append(label + "__E2I")
                            tag2Previous.append(label + "__E2L")
                        while True:
                            valid1 = True
                            vT1 = 0
                            for ei in range(start1, end1):
                                if not tags[ei].__eq__('O'):
                                    valid1 = False
                                    break
                            if not valid1:
                                valid1 = True
                                vT1 = 1
                                for ei in range(start1, end1):
                                    if not tags[ei].__eq__(tag1Previous[ei - start1]):
                                        valid1 = False
                                        vT1 = 0
                                        break
                            valid2 = True
                            vT2 = 0
                            for ei in range(start2, end2):
                                if not tags[ei].__eq__('O'):
                                    valid2 = False
                                    break
                            if not valid2:
                                valid2 = True
                                vT2 = 1
                                for ei in range(start2, end2):
                                    if not tags[ei].__eq__(tag2Previous[ei - start2]):
                                        valid2 = False
                                        vT2 = 0
                                        break
                            if valid1 and valid2:
                                c3+=1
                                for ei in range(start2, end2):
                                    tags[ei] = tag2Previous[ei - start2]
                                for ei in range(start1, end1):
                                    tags[ei] = tag1Previous[ei - start1]
                                flag = 1
                                break
                            else:
                                start1 += len(sentTokens)
                                end1 += len(sentTokens)
                                start2 += len(sentTokens)
                                end2 += len(sentTokens)
                            if end2 > kk * len(sentTokens):
                                kk += 1
                                for ki in range(len(sentTokens)):
                                    tags.append('O')
        newsent = dict()
        newsent['tokens'] = sentTokens
        newsent['tags'] = tags
        newsent['lentags/lentokens'] = kk * flag
        newsent['len'] = len(sentTokens)
        train_json_file.write(json.dumps(newsent) + '\n')
        if not ii %1000:
            print("Generate data:",ii)
    train_json_file.close()
    print(c1,c2,c3)

def datakk(file0, file1, kk=1, isTrain=True):
    fread = open(file0, 'r')
    sentence = fread.readlines()
    fwrite = open(file1, 'w')
    ii = 0
    print('Origin Sentence:' + str(len(sentence)))
    for line in sentence:
        sent = json.loads(line.strip('\r\n'))
        tkk = sent['lentags/lentokens']
        tkk = len(sent['tags']) // len(sent['tokens'])
        lent = len(sent['tokens'])
        if kk == 1:
            for i in range(tkk):
                newsent = dict()
                newsent['tokens'] = sent['tokens']
                newsent['tags'] = sent['tags'][i * lent:i * lent + lent]
                ii += 1
                fwrite.write(json.dumps(newsent) + '\n')

        elif kk == 2:
            if tkk >= 2:
                for i in range(tkk):
                    for j in range(i + 1, tkk):
                        newsent = dict()
                        newsent['tokens'] = sent['tokens']
                        newsent['tags'] = sent['tags'][i * lent:i * lent + lent]
                        newsent['tags'].extend(sent['tags'][j * lent:j * lent + lent])
                        fwrite.write(json.dumps(newsent) + '\n')
                        ii += 1
            else:
                newsent = dict()
                newsent['tokens'] = sent['tokens']
                newsent['tags'] = sent['tags']
                newsent['tags'].extend(['O' for i in sent['tokens']])
                fwrite.write(json.dumps(newsent) + '\n')
                ii += 1
        elif kk == 3:
            for _ in range(kk - tkk):
                sent['tags'].extend(['O' for i in sent['tokens']])
                tkk = 3
            for i in range(tkk):
                for j in range(i + 1, tkk):
                    for k in range(j + 1, tkk):
                        newsent = dict()
                        newsent['tokens'] = sent['tokens']
                        newsent['tags'] = sent['tags'][i * lent:i * lent + lent]
                        newsent['tags'].extend(sent['tags'][j * lent:j * lent + lent])
                        newsent['tags'].extend(sent['tags'][k * lent:k * lent + lent])
                        fwrite.write(json.dumps(newsent) + '\n')
                        ii += 1
        elif kk == 4:
            for _ in range(kk - tkk):
                sent['tags'].extend(['O' for i in sent['tokens']])
                tkk = 4
            for i in range(tkk):
                for j in range(i + 1, tkk):
                    for k in range(j + 1, tkk):
                        for m in range(k + 1, tkk):
                            newsent = dict()
                            newsent['tokens'] = sent['tokens']
                            newsent['tags'] = sent['tags'][i * lent:i * lent + lent]
                            newsent['tags'].extend(sent['tags'][j * lent:j * lent + lent])
                            newsent['tags'].extend(sent['tags'][k * lent:k * lent + lent])
                            newsent['tags'].extend(sent['tags'][m * lent:m * lent + lent])
                            fwrite.write(json.dumps(newsent) + '\n')
                            ii += 1
        elif kk == 8:
            newsent = dict()
            newsent['tokens'] = sent['tokens']
            newsent['tags'] = sent['tags']
            for j in range(kk - tkk):
                newsent['tags'].extend(['O' for i in sent['tokens']])
            fwrite.write(json.dumps(newsent) + '\n')
            ii += 1
    fread.close()
    fwrite.close()
    print("lenght of " + file1 + ' = ' + str(ii))




def make_idx_data_index_EE_LSTM(file, source_vob, max_s, target_vob,is_train=True):
    with open(file,"r") as f:
        maxlen=max([int(json.loads(fr)["len"]) for fr in f.readlines()])

    max_s = maxlen if maxlen<max_s or not is_train else max_s
    data_s_all = []
    data_t_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        len_s = len(sent['tokens'])
        t_sent = sent['tags']
        kk = len(sent['tags']) // len(sent['tokens'])
        max_t = kk * max_s
        data_t = []
        data_s = []
        if len(s_sent) > max_s:
            i = max_s - 1
            while i >= 0:
                ds = source_vob[s_sent[i]] if s_sent[i] in source_vob else source_vob["[unused50]"]
                data_s.append(ds)
                i -= 1
        else:
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)
            i = len(s_sent) - 1
            while i >= 0:
                ds=source_vob[s_sent[i]] if s_sent[i] in source_vob else source_vob["[unused50]"]
                data_s.append(ds)
                i -= 1
        data_s_all.append(data_s)
        if len(t_sent) > max_t:
            for i in range(kk):
                for word in t_sent[i * len_s:i * len_s + max_s]:
                    data_t.append(target_vob[word])
        else:
            for ki in range(kk):
                for word in t_sent[ki * len_s:ki * len_s + len_s]:
                    data_t.append(target_vob[word])
                for word in range(len_s, max_s):
                    data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all, data_t_all], max_s


def get_tag_index(labeltxt):

    target_vob = {}

    target_idex_word = {}
    f = open(labeltxt, 'r')
    fr = f.readlines()
    end = ['__E1S', '__E1B', '__E1I', '__E1L', '__E2S', '__E2B', '__E2I', '__E2L']
    count = 2
    target_vob['O'] = 1
    target_idex_word[1] = 'O'
    for line in fr:
        line = line.strip('\n')
        if line and not line == 'O':
            for lend in end:
                target_vob[line + lend] = count
                target_idex_word[count] = line + lend
                count += 1

    return target_vob, target_idex_word


def get_data_e2e(trainfile, testfile, labeltxt, source_vob, eelstmfile, maxlen=50):
    target_vob, target_idex_word= get_tag_index(labeltxt)
    print("source vocab size: " + str(len(source_vob)))
    print("target vocab size: " + str(len(target_vob)))
    # print("word2vec loaded!")
    #
    # if max_s > maxlen:
    #     max_s = maxlen
    # max_s = max(max_s, maxlen)
    # print('max soure sent lenth is ' + str(max_s))
    sourc_idex_word=dict(map(lambda x:(x[1],x[0]),source_vob.items()))
    train, maxtrainlen = make_idx_data_index_EE_LSTM(trainfile, source_vob, maxlen, target_vob)
    test, maxtestlen = make_idx_data_index_EE_LSTM(testfile, source_vob, maxlen, target_vob)
    print("max_train_word_len_sent:" + str(maxtrainlen))
    print("max_test_word_len_sent:" + str(maxtestlen))
    print("dataset created!")
    cPickle.dump([train, test,  [],source_vob, sourc_idex_word, target_vob, target_idex_word, maxtrainlen], open(eelstmfile, 'wb'))



if __name__=="__main__":
    dict_path = "/home/newdisk/syh/git_code/bert_dump/vocab.txt"
    token_dict = {}
    with open(dict_path, 'r') as reader:  # ,python3-> encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    input = "/home/newdisk/syh/Code/BS_new/data/Webnlg_new/"
    tag_sent(input+"train.json", input+"webnlg/train_bert.json", input+"label.txt", tokenizer)
    tag_sent(input + "test.json", input + "webnlg/test_bert.json", input + "label.txt", tokenizer)
    get_data_e2e(input+"webnlg/train_bert.json", input + "webnlg/test_bert.json", \
                 input+"label.txt", token_dict, input+"bert_e2edata.pkl", maxlen=100)
