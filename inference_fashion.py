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

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def search_item_from_jsom(json_file):
    with open(json_file, "r", encoding="utf8") as st_json:
        item_json = json.load(st_json)
    
    data = preprocessing_json(item_json)
    
    temp = [value for key,value in data.items()]    
    temp = [i for i in temp if i != 'None']
    temp = temp[0:-6]
    
    temp = flatten(temp)
    
    return temp
    
def inference_att(input_list,model_loaded,images_df_meta,top_i):
    new_vector  = model_loaded.infer_vector(input_list)
    sims = model_loaded.docvecs.most_similar([new_vector])
    tok_k = [int(i[0].split('_')[1]) for i in sims]
    plt.imshow(display_image(images_df_meta,tok_k[top_i]))

 
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="inference doc2vec for image meta recommendation")
    parser.add_argument(
            "--json", type=str, default='./data/example_jsonfile.json',
            help="json file path"
        )
    parser.add_argument(
            "--metadata", type=str, default='./data/images_df_meta_filter_format2.json',
            help="metadata file path"
        )
    parser.add_argument(
            "--model", type=str, default='./data/doc2vec_using_itemsplitidx_ustyle9',
            help="model weight file path"
        )
    parser.add_argument(
            "--topk", type=int, default=5,
            help="output top k"
        )
    
    args, _ = parser.parse_known_args()
    
    meta_path = args.metadata
    images_df_meta = pd.read_json(meta_path,orient='table')
    input_json_file = args.json
    
    input_list = search_item_from_jsom(input_json_file)
    
    model_path = args.model
    model_loaded = models.Doc2Vec.load(model_path)
   
    #input_list = ['니트웨어','노멀','섹시','타이트','단추']
    topk = args.topk
    
    inference_att(input_list,model_loaded,images_df_meta,topk)   
    