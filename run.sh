######
source activate py27
###Use cuda10.0 for tensorflow2.0
export PATH=/usr/local/cuda100/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda100/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 1）郑孙聪那个+固定word2vec输入，跑一下kbp原始数据集，
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=/home/newdisk/syh/Code/BS_new/data/KBP_old/ \
            -output=./result/KBP_old/ -e=./data/KBP_old/e2edata_word.pkl \
            -modelname=Zheng -emb=word -embtrainable=1 -batch=64 | tee ./result/KBP_old/Z_w_ko.out


# 2）郑孙聪那个+固定的bert向量，跑一下NYT，kbp以及webnlg的数据集，nyt和kbp都是原始的，webnlg用你处理过的就好，
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/KBP_old/ \
            -output=./result/KBP_old/ \
            -e=./data/KBP_old/e2edata_bert.pkl \
            -modelname=Zheng -emb=bert -embtrainable=0 -batch=64 | tee ./result/KBP_old/Z_b_ko.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/NYT_old/ \
            -output=./result/NYT_old/ \
            -e=./data/NYT_old/e2edata_bert.pkl \
            -modelname=Zheng -emb=bert -embtrainable=0 -batch=64 | tee ./result/NYT_old/Z_b_no.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/Webnlg_new/ \
            -output=./result/Webnlg_new/ \
            -e=./data/Webnlg_new/e2edata_bert.pkl \
            -modelname=Zheng -emb=bert -embtrainable=0 -batch=64 | tee ./result/Webnlg_new/Z_b_wn.out
# 3）你的ijcnn（词向量是word2vec）在webnlg上的实验，
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/Webnlg_new/ \
            -output=./result/Webnlg_new/ \
            -e=./data/Webnlg_new/e2edata_word.pkl \
            -modelname=Ijcnn -emb=word -embtrainable=0 -batch=64 | tee ./result/Webnlg_new/I_w_wn.out
# 4）你的ijcnn（词向量是bert）在nyt，kbp，以及webnlg上的实验（其中，加一组在nyt，kbp新数据集上实验）
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/KBP_old/ \
            -output=./result/KBP_old/ \
            -e=./data/KBP_old/e2edata_bert.pkl \
            -modelname=Ijcnn -emb=bert -embtrainable=0 -batch=64  | tee ./result/KBP_old/I_b_ko.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/NYT_old/ \
            -output=./result/NYT_old/ \
            -e=./data/NYT_old/e2edata_bert.pkl \
            -modelname=Ijcnn -emb=bert -embtrainable=0 -batch=64  | tee ./result/NYT_old/I_b_no.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/KBP_new/ \
            -output=./result/KBP_new/ \
            -e=./data/KBP_new/e2edata_bert.pkl \
            -modelname=Ijcnn -emb=bert -embtrainable=0 -batch=64  | tee ./result/KBP_new/I_b_kn.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/NYT_new/ \
            -output=./result/NYT_new/ \
            -e=./data/NYT_new/e2edata_bert.pkl \
            -modelname=Ijcnn -emb=bert -embtrainable=0 -batch=64  | tee ./result/NYT_new/I_b_nn.out
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=0 python ./MBert1024.py \
            -input=./data/Webnlg_new/ \
            -output=./result/Webnlg_new/ \
            -e=./data/Webnlg_new/e2edata_bert.pkl \
            -modelname=Ijcnn -emb=bert -embtrainable=0 -batch=64  | tee ./result/Webnlg_new/I_b_wn.out
