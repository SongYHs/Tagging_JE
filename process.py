try:
    import pickle as cPickle    #python3
except:
    import cPickle              #python2
import json,nltk,unicodedata
import numpy as np

def tag_sent(raw_file,new_file,labelfile,max_token=50):
    train_json_file = open(new_file, 'w')
    file = open(raw_file, 'r')
    sentences_0 = file.readlines()
    c = 0
    Tkk = []
    ii = 0
    Tkk = {}
    vV = []
    Mlabel = {}
    Alabel = {}
    count_r = {}
    label = []
    f = open(labelfile, 'r')
    fr = f.readlines()
    label = [line.strip('\n') for line in fr]
    f.close()
    for line in sentences_0:
        c += 1
        kk = 1
        count_r[c - 1] = 0
        Tkk[c - 1] = 0
        if not c % 10000:
            print(c)
        sent = json.loads(line.strip('\r\n'))
        flag = 0
        sentText = str(unicodedata.normalize('NFKD', sent['sentText']).encode('ascii', 'ignore')).rstrip('\n').rstrip('\r')
        #sentText=sent['sentText'] python3
        tokens = nltk.word_tokenize(sentText)
        tags=["O"]*len(tokens)
        emIndexByText = {}
        for em in sent['entityMentions']:
            emText = unicodedata.normalize('NFKD', em['text']).encode('ascii', 'ignore')
            # emText=em['text']
            tokens1 = tokens
            em1 = emText.split()
            flagE = True
            if emIndexByText.__contains__(emText):
                flagE = False
            while flagE:
                start, end = find_index(tokens1, em1)
                if start != -1 and end != -1:
                    tokens1 = tokens1[end:]
                    if emText not in emIndexByText:
                        emIndexByText[emText] = [(start, end)]
                    elif not emIndexByText[emText].__contains__((start, end)):
                        offset = emIndexByText[emText][-1][1]
                        emIndexByText[emText].append((start + offset, end + offset))
                else:
                    break
        for rm in sent['relationMentions']:
            if not rm['label'].__eq__('None') and label.__contains__(rm['label']):
                rmlabel = rm["label"]
                if not Alabel.__contains__(rmlabel):
                    Alabel[rmlabel] = [c - 1]
                else:
                    Alabel[rmlabel].append(c - 1)
                em1 = unicodedata.normalize('NFKD', rm['em1Text']).encode('ascii', 'ignore')
                em2 = unicodedata.normalize('NFKD', rm['em2Text']).encode('ascii', 'ignore')
                # em1 = rm["em1Text"] #python3
                # em2=rm['em2Text'] #python3
                if emIndexByText.__contains__(em1) and emIndexByText.__contains__(em2):
                    ind1 = emIndexByText[em1]
                    ind2 = emIndexByText[em2]
                    minind = len(tokens)
                    labelindex = []
                    for i1ind, i1 in enumerate(ind1):
                        for i2ind, i2 in enumerate(ind2):
                            if (i2[0] - i1[1]) * (i2[1] - i1[0]) > 0:
                                if minind > abs(i2[1] - i1[1]):
                                    minind = abs(i2[1] - i1[1])
                                    labelindex = [i1ind, i2ind]
                    if labelindex:
                        i1ind = labelindex[0]
                        i2ind = labelindex[1]
                        start1 = ind1[i1ind][0]
                        end1 = ind1[i1ind][1]
                        start2 = ind2[i2ind][0]
                        end2 = ind2[i2ind][1]
                        tag1Previous = []
                        tag2Previous = []
                        if end1 - start1 == 1:
                            tag1Previous.append(rmlabel + "__E1S")
                        elif end1 - start1 == 2:
                            tag1Previous.append(rmlabel + "__E1B")
                            tag1Previous.append(rmlabel + "__E1L")
                        else:
                            tag1Previous.append(rmlabel + "__E1B")
                            for ei in range(start1 + 1, end1 - 1):
                                tag1Previous.append(rmlabel + "__E1I")
                            tag1Previous.append(rmlabel + "__E1L")
                        if end2 - start2 == 1:
                            tag2Previous.append(rmlabel + "__E2S")
                        elif end2 - start2 == 2:
                            tag2Previous.append(rmlabel + "__E2B")
                            tag2Previous.append(rmlabel + "__E2L")
                        else:
                            tag2Previous.append(rmlabel + "__E2B")
                            for ei in range(start2 + 1, end2 - 1):
                                tag2Previous.append(rmlabel + "__E2I")
                            tag2Previous.append(rmlabel + "__E2L")
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
                                for ei in range(start2, end2):
                                    tags[ei] = tag2Previous[ei - start2]
                                for ei in range(start1, end1):
                                    tags[ei] = tag1Previous[ei - start1]
                                Tkk[c - 1] = kk
                                if not (vT1 and vT2):
                                    ii += 1
                                    count_r[c - 1] += 1
                                    if not Mlabel.__contains__(rmlabel):
                                        Mlabel[rmlabel] = [c - 1]
                                    else:
                                        Mlabel[rmlabel].append(c - 1)
                                flag = 1
                                if (vT1 or vT2) and not (vT1 and vT2):
                                    vV.append(c - 1)
                                break
                            else:
                                start1 += len(tokens)
                                end1 += len(tokens)
                                start2 += len(tokens)
                                end2 += len(tokens)
                            if end2 > kk * len(tokens):
                                kk += 1
                                for ki in range(len(tokens)):
                                    tags.append('O')
        newsent = dict()
        newsent['tokens'] = tokens
        newsent['tags'] = tags
        newsent['lentags/lentokens'] = kk * flag
        train_json_file.write(json.dumps(newsent) + '\n')
    train_json_file.close()
    return Tkk, vV, ii, Alabel, count_r, Mlabel
def find_index(sen_split, word_split):
    index1 = -1
    index2 = -1
    for i in range(len(sen_split)):
        if str(sen_split[i]) == str(word_split[0]):
            flag = True
            k = i
            for j in range(len(word_split)):
                if str(word_split[j])!= sen_split[k]:
                    flag = False
                if k < len(sen_split) - 1:
                    k+=1
            if flag:
                index1 = i
                index2 = i + len(word_split)
                break
    return index1, index2

def load_vec_pkl(fname,vocab,k=100):
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = cPickle.load(open(fname,'rb'))#,encoding='iso-8859-1')
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    i=0
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
            i+=1
        W[vocab[word]] = w2v[word]
    print(str(i)+' words unknow in pretrain words')
    return w2v,k,W
def make_idx_data_index_EE_LSTM(file,source_vob,target_vob,max_s=50):
    data_s_all=[]
    data_t_all=[]
    length=[]
    kk=[]
    f = open(file,'r')
    fr = f.readlines()
#    sent=json.loads(fr[0].strip('\r\n'))

    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        len_s=len(sent['tokens'])
        t_sent = sent['tags']
        data_s=[source_vob[si] for si in s_sent][::-1]
        data_s_all.append(data_s)
        data_t=[target_vob[word] for word in t_sent]
        data_t_all.append(data_t)
        length.append(len_s)
        kk.append(len(t_sent)//len_s)
    f.close()
    return [data_s_all,data_t_all,length,kk]
def make_idx_data_index_EE_LSTM1(file,source_vob,target_vob,max_s=50):
    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    sent=json.loads(fr[0].strip('\r\n'))
    kks=[]
    length=[]
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        len_s=len(sent['tokens'])
        t_sent = sent['tags']
        kk=len(sent['tags'])//len(sent['tokens'])
        kks.append(kk)
        max_t=kk*max_s
        data_t = []
        data_s = []
        length.append(max_s)
        if len(s_sent) > max_s:
            i=max_s-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        else:
            num=max_s-len(s_sent)
            for inum in range(0,num):
                data_s.append(0)
            i=len(s_sent)-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        data_s_all.append(data_s)
        if len(t_sent) > max_t:
            for i in range(kk):
                for word in t_sent[i*len_s:i*len_s+max_s]:
                    data_t.append(target_vob[word])
        else:
            for ki in range(kk):
                for word in t_sent[ki*len_s:ki*len_s+len_s]:
                    data_t.append(target_vob[word])
                for word in range(len_s,max_s):
                    data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all,data_t_all,length,kks]


def get_word_index(train,test,labeltxt):
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    f=open(labeltxt,'r')
    fr=f.readlines()
    end=['__E1S','__E1B','__E1I','__E1L','__E2S','__E2B','__E2I','__E2L']
    count = 2
    target_vob['O']=1
    target_idex_word[1]='O'
    for line in fr:
        line=line.strip('\n')
        if line and not line=='O':
            for lend in end:
                target_vob[line+lend]=count
                target_idex_word[count]=line+lend
                count+=1
    count = 1
    max_s=0
    f.close()
    f = open(train,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()
    f.close()
    f = open(test,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()
    f.close()
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count+=1
    if not source_vob.__contains__("UNK"):
        source_vob["UNK"] = count
        sourc_idex_word[count] = "UNK"
        count+=1
    return source_vob,sourc_idex_word,target_vob,target_idex_word,max_s
def get_data_e2e(trainfile,testfile,labeltxt,w2v_file,eelstmfile,maxlen=50):
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s0 = \
                get_word_index(trainfile, testfile,labeltxt)
    print("source vocab size: " + str(len(source_vob)))
    print("target vocab size: " + str(len(target_vob)))
    if trainfile.__contains__('webnlg'):
        k0=100
    else:
        k0=300
    source_w2v ,k ,source_W= load_vec_pkl(w2v_file,source_vob,k=k0)
    print("word2vec loaded!")
    print("num words in source word2vec: " + str(len(source_w2v))+\
          "source  unknown words: "+str(len(source_vob)-len(source_w2v)))

    max_s = maxlen if max_s0 > maxlen else max_s0
    print('max soure sent lenth is ' + str(max_s)+"other"+str(max_s0))
    train = make_idx_data_index_EE_LSTM1(trainfile,source_vob,target_vob,max_s=max_s)
    test = make_idx_data_index_EE_LSTM1(testfile, source_vob, target_vob,max_s=max_s)
    print("dataset created!")
    cPickle.dump([train,test,source_W,source_vob,sourc_idex_word,
                  target_vob,target_idex_word,max_s],open(eelstmfile,'wb'))



if __name__=="__main__":
    # raw_train="/home/newdisk/Code/BS_new/data/NYT_old/train.json"
    # raw_test="/home/newdisk/Code/BS_new/data/NYT_old/test.json"
    # seq_train="/home/newdisk/Code/BS_new/data/NYT_old/train_0926.json"
    # seq_test="/home/newdisk/Code/BS_new/data/NYT_old/test_0926.json"
    # labelfile="/home/newdisk/Code/BS_new/data/NYT_old/label.txt"
    # w2vfile="/home/newdisk/Code/BS_new/data/NYT_old/w2v.pkl"
    # e2edatafile="/home/newdisk/Code/BS_new/data/NYT_old/e2edata_09262_pycharme.pkl"
    # tag_sent(raw_train,seq_train,labelfile)
    # tag_sent(raw_test,seq_test,labelfile)
    # get_data_e2e(seq_train, seq_test, labelfile, w2vfile, e2edatafile)

    raw_train = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/train.json"
    raw_test = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/test.json"
    seq_train = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/train_0926.json"
    seq_test = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/test_0926.json"
    labelfile = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/label.txt"
    w2vfile = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/w2v.pkl"
    e2edatafile = "/home/newdisk/Code/BS_new/data/NYT_reshuffle/e2edata_09261_Ding.pkl"
    tag_sent(raw_train, seq_train, labelfile)
    tag_sent(raw_test, seq_test, labelfile)
    get_data_e2e(seq_train, seq_test, labelfile, w2vfile, e2edatafile)


""""""
