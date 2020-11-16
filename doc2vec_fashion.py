# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:34:08 2020

@author: Lee jung ho
"""

from gensim.test.utils import datapath
from gensim import utils
import random
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec 
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim import models
import gensim.models
from gensim import models

import shutil
from tqdm import tqdm
import numpy as np
#data_path = os.path.join('./K-fashion(2)/K-fashion(2)')
import os, shutil
import json
from glob import glob

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import json
import itertools
import collections
import matplotlib.pyplot as plt

from utils.utils import column_one_hot ,cos_sim, copytree, check_zero,display_image,search_best_neighbor_gt
from utils.utils import search_best_neighbor,column_split
from utils.utils import plot_2d_graph,search_best_neighbor_gt_topk,preprocessing_json


from distutils.dir_util import copy_tree
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
font_list = font_manager.findSystemFonts(fontpaths="c:/Windows/Fonts", fontext='ttf')

rc('font', family=font_name)

print('# 설정 되어있는 폰트 사이즈')
print (plt.rcParams['font.size'] ) 
print('# 설정 되어있는 폰트 글꼴')
print (plt.rcParams['font.family'] )
plt.rcParams["font.family"] = 'malgun.ttf'
import matplotlib.pyplot as plt 

import gensim.models
from gensim import models

class LabeledLineSentence(object):
    def __init__(self, item_corpus,suffle = True):
        self.data = item_corpus
        self.suffle = suffle
    def __iter__(self):
        for uid, line in enumerate(self.data):
            if self.suffle :
                random.shuffle(line)
            print(line,"\n")
            yield models.doc2vec.LabeledSentence(words=line, tags=['item_%s' % uid])

class LabeledLineSentenceByAttribute(object):
    def __init__(self, item_corpus,item_index,suffle = True):
        self.data = item_corpus
        self.suffle = suffle
        self.item_index = item_index
    def __iter__(self):
        for idx,(line, i_idx) in enumerate(zip(self.data,self.item_index)):
            if self.suffle :
                random.shuffle(line)
            print(line,"\n")
            yield models.doc2vec.LabeledSentence(words=line, tags=['item_%s' % i_idx])


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self,item_corpus) :
        self.data = item_corpus
    def __iter__(self):
        
        for line in self.data  :
            # assume there's one document per line, tokens separated by whitespace
            random.shuffle(line)
            print(line)
            yield line

class callback(CallbackAny2Vec): 
    """Callback to print loss after each epoch.""" 
    def __init__(self): 
        self.epoch = 0 
        self.loss_to_be_subed = 0 
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss() 
        loss_now = loss - self.loss_to_be_subed 
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now)) 
        self.epoch += 1

def drop_none(table) :
        
    tt = [[x for x in line if x !='None' ] for line in table]
    tt = [[x for x in line if x != None ] for line in tt]
    return tt
    
if __name__=='__main__':
    images_df_meta = pd.read_json('images_df_meta_filter_format2.json',orient='table')
    
    images_df_using = pd.read_json('images_df_using_filter_format2.json',orient='table')
    
    temp_c = images_df_using.columns 
  
    up_c_idx = [i for i,k in enumerate(temp_c) if k.split('_')[0] == '상의']
    down_c_idx = [i for i,k in enumerate(temp_c) if k.split('_')[0] == '하의']
    onp_c_idx = [i for i,k in enumerate(temp_c) if k.split('_')[0] == '원피스']
    out_c_idx = [i for i,k in enumerate(temp_c) if k.split('_')[0] == '아우터']
    sty_c_idx = [i for i,k in enumerate(temp_c) if k == '스타일']
    substy_c_idx = [i for i,k in enumerate(temp_c) if k == '서브스타일']
    sty_c_idx.extend(substy_c_idx)

    images_df_up_using = images_df_using.iloc[:,up_c_idx]
    images_df_down_using = images_df_using.iloc[:,down_c_idx]
    images_df_onp_using = images_df_using.iloc[:,onp_c_idx]
    images_df_out_using = images_df_using.iloc[:,out_c_idx]
    images_df_sty_using = images_df_using.iloc[:,sty_c_idx]

    a = images_df_up_using.to_numpy()
    b = images_df_down_using.to_numpy()
    c = images_df_onp_using.to_numpy()
    d = images_df_out_using.to_numpy()
    e = images_df_sty_using.to_numpy()

    a = a.tolist()
    b = b.tolist()
    c = c.tolist()
    d = d.tolist()
    e = e.tolist()


    #이거는 각 아이템에 해당하는 속성을 부여하는 방식인데, 일단 none 과 'none' 통일 시킨다음에 진행해야함
    a = drop_none(a)
    b = drop_none(b)
    c = drop_none(c)
    d = drop_none(d)
    e = drop_none(e)

    [line if len(line) == 0 else line.extend(sty) for line,sty in zip(a,e)]
    [line if len(line) == 0 else line.extend(sty) for line,sty in zip(b,e)]
    [line if len(line) == 0 else line.extend(sty) for line,sty in zip(c,e)]
    [line if len(line) == 0 else line.extend(sty) for line,sty in zip(d,e)]

    a.extend(b)
    a.extend(c)
    a.extend(d)
    #a.extend(e)

    total_input = a
    
    item_index = np.arange(0,len(images_df_up_using)*4)
    item_index = item_index%len(images_df_up_using)

    input_x = images_df_using.to_numpy()

    total_input = drop_none(total_input)
    #len(0) 삭제
    use_input = [(line,i_idx) for line,i_idx in zip(total_input,item_index) if len(line)!=0] 

    input_d2v_x = [line for line,idx in use_input]
    item_index = [idx for line,idx in use_input]

    using_att_idx=drop_none(images_df_using.to_numpy())   
   
    model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
    sentences = LabeledLineSentenceByAttribute(input_d2v_x,item_index,suffle=True)

    model.build_vocab(sentences)
    model = models.Doc2Vec(
        documents=sentences, 
        min_count=1, size=50,
        window=1,
        iter=30,
        workers=10,
        #callbacks=[callback()]
        )  

    model.save("doc2vec_using_itemsplitidx_ustyle9")
    #ver2 : extend each item to att 
    model_loaded = models.Doc2Vec.load('doc2vec_using_itemsplitidx_ustyle9')

    #print (model_loaded.docvecs.most_similar(["item_0"]))
    #model_loaded.docvecs.most_similar(["item_39"])
    #model['item_0']

    #model_loaded.wv.most_similar(['티셔츠'])
    
    gt_pair = {}
    total_item_num = len(input_x)
    for i in tqdm(range(total_item_num)) :
        try :
            topk_list = model_loaded.docvecs.most_similar(["item_{}".format(i)],topn=20)[0:]
            topk_list = [item.split('_')[1] for item,score in topk_list]
            gt_pair[i] = (topk_list)
            
            # pair_items = model_loaded.docvecs.most_similar(["item_{}".format(i)])[0][0]
            # pair_idx = pair_items.split('_')[1]
            # gt_pair[i] = str(pair_idx)
            
        except :
            print ("only havs style")
            #gt_pair[i] = 0
            continue
    
    with open("./gt_pair/d2v_suffle_none_pred_pair_itemsplit_usestyle9_top20.json", "w") as json_file:
        json.dump(gt_pair, json_file)

    with open("./gt_pair/d2v_suffle_none_pred_pair_itemsplit_usestyle9_top20.json", "r") as json_file:
        gt_pair = json.load(json_file)
    
    for i in range(500,510) :
    search_best_neighbor_gt_topk(images_df_meta,images_df_using,using_att_idx,i,gt_pair,\
                            'd2v_suffle_none_pred_pair_itemsplit_usestyle9_top20',save=True,topn=20)

    