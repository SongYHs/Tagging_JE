# Tagging_JE
基于标记策略的联合抽取

## 数据包括
 1.NYT，KBP，Webnlg数据集，
 2.采用预训练的模型需要下载相应的参数w2v(),和bert()
 3.使用nltk需要下载()
## 环境要求
  python=2.7
  tensorflow>=1.14
  keras=2.3.1
  nltk=3.4
## 运行
 source run.sh  #运行环境已被命名为py27
 ### 可选参数为：
            -input=./data/KBP_old/             \原始数据
            -output=./result/KBP_old/          \模型输出
            -e=./data/KBP_old/e2edata_word.pkl \对处理好的数据命名
            -modelname=Zheng                   \可选【Zheng,Ijcnn】,采用模型为Zheng, et al. 2017 还是Song et al 2019
            -emb=word                          \采用bert词嵌入还是word2vec
            -embtrainable=1                    \是否固定词嵌入参数
            -batch=64                          
            -epoch=50
 
