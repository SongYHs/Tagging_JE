#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:20:51 2019

@author: Song Yunhua
"""

"""        Triplets             """
def evaluavtion_triple_1(testresult,max_s=50,kk=1):
    total_predict_right=0.
    total_predict=0.
    total_right = 0.
    prn={}
    p={}
    r={}
    rpn={}
#    print(len(testresult))
    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum ,rightnum,prn0,p0,r0,rpn0= count_sentence_triple_num(ptag,ttag,max_s)
        total_predict_right+=predictrightnum
        total_predict+=predictnum
        total_right += rightnum
        for i in range(kk):
            if not p.__contains__(i):
                p[i]=p0[i]
            else:
                p[i]+=p0[i]
            if not rpn.__contains__(i):
                rpn[i]=rpn0[i]
            else:
                rpn[i]+=rpn0[i]
        for i in range(len(r0)):
            if not r.__contains__(i):
                r[i]=r0[i]
            else:
                r[i]+=r0[i]
            if not prn.__contains__(i):
                prn[i]=prn0[i]
            else:
                prn[i]+=prn0[i]
    print(total_predict_right,total_predict,total_right,rpn,p,prn,r)    
    P = total_predict_right /float(total_predict) if total_predict!=0 else 0
    R = total_predict_right /float(total_right)
    F = (2*P*R)/float(P+R) if P!=0 else 0
    return P,R,F,total_predict_right,total_predict,total_right,prn,p,r

def evaluavtion_triple_Ki(testresult,max_s=50):
    
    nprf={}
    for ii,sent in enumerate(testresult):
        ptag=sent[0]
        ttag=sent[1]
        k=len(ttag)//max_s
        predictrightnum, predictnum ,rightnum,prn0,p0,r0,rpn0= count_sentence_triple_num(ptag,ttag,max_s)
        if nprf.__contains__(k):
            nprf[k][0]+=predictrightnum
            nprf[k][1]+=predictnum
            nprf[k][2]+=rightnum
            nprf[k][3]+=1
        else:
            nprf[k]=[predictrightnum,predictnum,rightnum,1]
    PRF={}
    ap=0
    ar=0
    apr=0
    for k in nprf:
        apr+=nprf[k][0]
        ap+=nprf[k][1]
        ar+=nprf[k][2]
        P=nprf[k][0]/float(nprf[k][1]) if nprf[k][1]!=0 else 0
        R=nprf[k][0]/float(nprf[k][2])
        F=2*P*R/float(P+R) if P!=0 else 0
        PRF[k]={"P":P,"R":R,"F":F,"S_num":nprf[k][3],"R_num":nprf[k][1]}
        print("ki:",k," P:",P," R:",R," F:",F," S_num:",nprf[k][3]," R_num:",nprf[k][2],' PR_num:',nprf[k][0],' P_num:',nprf[k][1])
    P=apr/float(ap) if ap!=0 else 0
    R=apr/float(ar)
    F=2*P*R/float(P+R) if P!=0 else 0
    PRF[0]={"P":P,"R":R,"F":F,"S_num":len(testresult),"R_num":ar}
#    print("ki:",k," P:",P," R:",R," F:",F," S_num:",len(testresult)," R_num:",ar,' PR_num:',apr,' P_num:',ar)
    return PRF
    


def count_sentence_triple_num(ptag,ttag,max_s=50):
    #transfer the predicted tag sequence to triple index
    
    predict_triplet=[]
    right_triplet=[]
    predict={}
    rpn0={}
    prn0={}
    p0={}
    r0={}
    predict_right_num = 0       # the right number of predicted triple
    for i in range(len(ptag)//max_s):
        predict_rmpair0= tag_to_triple_index_new1(ptag[i*max_s:i*max_s+max_s])
        predict_triplet0=get_triplet(predict_rmpair0)
        for pt in predict_triplet0:
            if not predict_triplet.__contains__(pt):
                predict_triplet.append(pt)
        p0[i]=len(predict_triplet0)
        predict[i]=predict_triplet0
    for i in range(len(ttag)//max_s):
        right_rmpair0 = tag_to_triple_index_new1(ttag[i*max_s:i*max_s+max_s])
        right_triplet0=get_triplet(right_rmpair0)
        nprn=0
        for rt in right_triplet0:
            if not right_triplet.__contains__(rt):
                right_triplet.append(rt)
                if predict_triplet.__contains__(rt):
                    predict_right_num+=1
                    nprn+=1
        prn0[i]=nprn
        r0[i]=len(right_triplet0)
    
    predict_num = len(predict_triplet)     # the number of predicted triples
    right_num = len(right_triplet)
    for i in predict:
        predict_triplet0=predict[i]
        nrpn=0
        for pt in predict_triplet0:
            if right_triplet.__contains__(pt):
                nrpn+=1
        rpn0[i]=nrpn
  
    return predict_right_num,predict_num,right_num,prn0,p0,r0,rpn0
def get_triplet(predict_rmpair):
    triplet=[]
    for type1 in predict_rmpair:
        eelist = predict_rmpair[type1]
        e1 = eelist[0]
        e2 = eelist[1]
        if len(e2)<len(e1):
            for i in range(len(e2)):
                e2i0,e2i1=e2[i][0],e2[i][1]
                e2i=(e2i0+e2i1)/2
                mineij=100
                for j in range(len(e1)):
                    e1j=(e1[j][0]+e1[j][1])/2
                    if mineij>abs(e1j-e2i):
                        mineij=abs(e1j-e2i)
                        e1ij=e1[j]
                e1.remove(e1ij)
                triplet.append((e1ij,type1,e2[i]))
        else:
            for i in range(len(e1)):
                e1i0,e1i1=e1[i][0],e1[i][1]
                e1i=(e1i0+e1i1)/2
                mineij=100
                for j in range(len(e2)):
                    e2j=(e2[j][0]+e2[j][1])/2
                    if mineij>abs(e1i-e2j):
                        mineij=abs(e1i-e2j)
                        e2ij=e2[j]
                e2.remove(e2ij)
                triplet.append((e1[i],type1,e2ij))
    return triplet
"""        entity             """
def evaluavtion_entity(testresult,kk=1,max_s=50):
    tpr1=0.
    tp1= 0.
    tr1= 0.
    
    tpr2=0.
    tp2= 0.
    tr2= 0.
    
    tprs=0.
    tps= 0.
    trs= 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        pr1,p1,r1,pr2,p2,r2,prs,ps,rs= count_sentence_entitys_num(ptag,ttag,max_s)
        
        tpr1+=pr1
        tp1 += p1
        tr1 += r1
        
        tpr2+=pr2
        tp2 += p2
        tr2 += r2
        
        tprs+=prs
        tps += ps
        trs += rs
        
    P1 = tpr1 /float(tp1) if tp1!=0 else 0
    R1 = tpr1 /float(tr1)
    F1 = (2*P1*R1)/float(P1+R1) if P1!=0 else 0
    
    P2 = tpr2 /float(tp2) if tp2!=0 else 0
    R2 = tpr2 /float(tr2)
    F2 = (2*P2*R2)/float(P2+R2) if P2!=0 else 0
    
    Ps = tprs /float(tps) if tps!=0 else 0
    Rs = tprs /float(trs)
    Fs = (2*Ps*Rs)/float(Ps+Rs) if Ps!=0 else 0
    
    return P1,R1,F1,P2,R2,F2,Ps,Rs,Fs
def count_sentence_entitys_num(ptag,ttag,max_s=50):
    #transfer the predicted tag sequence to triple index
    
    pe1s=[]
    pe2s=[]
    pess=[]
    re1s=[]
    re2s=[]
    ress=[]
    

    for i in range(len(ttag)//max_s):
        predict_rmpair0= tag_to_triple_index_new1(ptag[i*max_s:i*max_s+max_s])
        pe1,pe2,pes=get_entity(predict_rmpair0)
        for pt in pe1:
            if not pe1s.__contains__(pt):
                pe1s.append(pt)
            if not pess.__contains__(pt):
                pess.append(pt)
        for pt in pe2:
            if not pe2s.__contains__(pt):
                pe2s.append(pt)
            if not pess.__contains__(pt):
                pess.append(pt)
    for i in range(len(ttag)//max_s):
        right_rmpair0 = tag_to_triple_index_new1(ttag[i*max_s:i*max_s+max_s])
        re1,re2,res=get_entity(right_rmpair0)
        for pt in re1:
            if not re1s.__contains__(pt):
                re1s.append(pt)
            if not ress.__contains__(pt):
                ress.append(pt)
        for pt in re2:
            if not re2s.__contains__(pt):
                re2s.append(pt)
            if not ress.__contains__(pt):
                ress.append(pt)
    pr1 = 0       # the right number of predicted triple
    p1 = len(pe1s)     # the number of predicted triples
    r1 = len(re1s)
    for triplet in pe1s:
        if re1s.__contains__(triplet):
            pr1+=1 
    pr2 = 0       # the right number of predicted triple
    p2 = len(pe2s)     # the number of predicted triples
    r2 = len(re2s)
    for triplet in pe2s:
        if re2s.__contains__(triplet):
            pr2+=1  
    prs = 0       # the right number of predicted triple
    ps = len(pess)     # the number of predicted triples
    rs = len(ress)
    for triplet in pess:
        if ress.__contains__(triplet):
            prs+=1  
    return pr1,p1,r1,pr2,p2,r2,prs,ps,rs
def get_entity(predict_rmpair):
    entity1=[]
    entity2=[]
    entity_all=[]
    for type1 in predict_rmpair:
        e1=predict_rmpair[type1][0]
        e2=predict_rmpair[type1][1]
        for e1i in e1:
            if not entity1.__contains__(e1i):
                entity1.append(e1i)
            if not entity_all.__contains__(e1i):
                entity_all.append(e1i)
        for e1i in e2:
            if not entity2.__contains__(e1i):
                entity2.append(e1i)
            if not entity_all.__contains__(e1i):
                entity_all.append(e1i)
    return entity1,entity2,entity_all
                

def tag_to_triple_index_new1(ptag):
    rmpair={}
    for i in range(0,len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist=[]
                e1=[]
                e2=[]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
#                            print(j,len(ptag))
                            if ptag[j].__contains__("1") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist=rmpair[type_e[0]]
                e1=eelist[0]
                e2=eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e2.append((i, j))
                eelist[0]=e1
                eelist[1]=e2
                rmpair[type_e[0]] = eelist
    return rmpair
def tag_to_triple(ptag,senttext):
    triplets=[]
    rmpair={}
    i=0
    for i in range(len(ptag)):
        tag=ptag[i]
        if tag.__contains__("__E1S"):
            labelt = tag.split("__")[0]
            if not rmpair.__contains__(labelt):
                rmpair[labelt]={}
                rmpair[labelt]['e1']=set()
                rmpair[labelt]['e2']=set()
                rmpair[labelt]['e1'].add(senttext[i])
            else:
                rmpair[labelt]['e1'].add(senttext[i])
        elif tag.__contains__("E2S"):
            labelt = tag.split("__")[0]
            if not rmpair.__contains__(labelt):
                rmpair[labelt]={}
                rmpair[labelt]['e1']=set()
                rmpair[labelt]['e2']=set()
                rmpair[labelt]['e2'].add(senttext[i])
            else:
                rmpair[labelt]['e2'].add(senttext[i])
        elif tag.__contains__("__E1B"):
            label0=tag.split("__")[0]
            j=i+1
            v='B'
            em=senttext[i]
            while j<len(ptag):
                tagj=ptag[j]
                if not tagj.__contains__(label0):
                    break
                elif tagj.__contains__("__E1B") or tagj.__contains__("__E2") or tagj.__contains__("__E1S"):
                    break
                elif tagj.__contains__("__E1I"):
                    v='I'
                    em+=' '+senttext[j]
                    j+=1
                elif tagj.__contains__("__E1L"):
                    v='L'
                    em+=' '+senttext[j]
                    break
            if v=='L':
                if not rmpair.__contains__(label0):
                    rmpair[label0]={}
                    rmpair[label0]['e1']=set()
                    rmpair[label0]['e2']=set()
                    rmpair[label0]['e1'].add(em)
                else:
                    rmpair[label0]['e1'].add(em)
        elif tag.__contains__("__E2B"):
            label0=tag.split("__")[0]
            j=i+1
            v='B'
            em=senttext[i]
            while j<len(ptag):
                tagj=ptag[j]
                if not tagj.__contains__(label0):
                    break
                elif tagj.__contains__("__E2B") or tagj.__contains__("__E1") or tagj.__contains__("__E2S"):
                    break
                elif tagj.__contains__("__E2I"):
                    v='I'
                    em+=' '+senttext[j]
                    j+=1
                elif tagj.__contains__("__E2L"):
                    v='L'
                    em+=' ' +senttext[j]
                    break
            if v=='L':
                if not rmpair.__contains__(label0):
                    rmpair[label0]={}
                    rmpair[label0]['e1']=set()
                    rmpair[label0]['e2']=set()
                    rmpair[label0]['e2'].add(em)
                else:
                    rmpair[label0]['e2'].add(em)
    for labelt in rmpair:
        for em1 in rmpair[labelt]['e1']:
            for em2 in rmpair[labelt]['e2']:
                triplet=em1+'____'+labelt+'____'+em2
                triplets.append(triplet)
    return triplets