# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:06:46 2020

@author: Lee jung ho
"""
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# =============================================================================
#  set directory for raw data (F:\k-fashion)
# =============================================================================
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

plt.rcParams["figure.figsize"] = (12,10)
plt.rcParams['figure.max_open_warning'] = 5


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        try :
            print(item)    
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
        except FileExistsError :
            continue 


def split_name(x):
    #x = '688A3236.JPG'
    if len(x.split('_')) >3 :
        try :
            int(x.split('_')[1])
            item_index = x.split('_')[1]
            item_order = x.split('_')[2]
        except ValueError :
            item_index = x.split('_')[2]
            item_order = x.split('_')[3]
            #item_index = x.split('_')[0:2]
            #item_index = '_'.join(item_index)
    #x = 'batch_REIGN_001_04.jpg'
    elif len(x.split('_')) == 3:
        if x[0] =='u' :
            item_index = x.split('_')[1][0:6]
            item_order = x.split('_')[2]
        else :
            item_index = x.split('_')[1]
            item_order = x.split('_')[2]
    elif len(x.split('_')) == 2 :
        try :
            int(x.split('_')[0])
            item_index = x.split('_')[0][0:6]
            item_order = x.split('_')[1]
        except ValueError :
            item_index = x.split('_')[0]
            item_order = x.split('_')[1]
    else :
       item_index = x.split('_')[0]
       item_order = ""
    return item_index,item_order

def index_check(x):
    #x = 'batch_REIGN_001_04.jpg'
    if len(x.split('_')) != 3:
        binary_use = False
    
    else :
        try : 
            int(x.split("_")[0])
            binary_use = False
        except ValueError :
            binary_use = True
    return binary_use

if __name__=='__main__':
    data_path1 = os.path.join('./K-fashion(1)/K-fashion(1)')
    copytree(os.path.join(data_path1),'./datasets_test')
    data_path2 = os.path.join('./K-fashion(2)/K-fashion(2)')
    copytree(os.path.join(data_path2),'./datasets_test')
    
    ## find image per item

    img_root = 'F:/k-fashion'

# ## 일단 데이터 한곳에 모으기 폴더 정리도 좀하고    
# from distutils.dir_util import copy_tree

# data_path = os.path.join('./K-fashion(2)/K-fashion(2)')
# copy_tree(os.path.join(data_path),'./datasets')
    data_path = os.path.join('./datasets')

    label_path = []
    img_path = []
    not_match_count = 0
    for folder_name in tqdm(os.listdir(data_path)) :
        #folder_name = os.listdir(data_path)[1]
        try :
            if len(os.listdir(os.path.join(data_path,folder_name))) != 2 :
                not_match_count +=1
            label_path += glob(os.path.join(data_path,folder_name,"*.json"))
            img_path += glob(os.path.join(data_path,folder_name,"*.jpg"))
        except NotADirectoryError :
            print('not data folder')
            continue
    print('not matched file num {}'.format(not_match_count))

        
    null_json_num = 0
    item_binary = []
    
    #total_up_keys = set()
    total_up_keys_num = []
    total_up_values_num = []
    
    total_down_keys_num = []
    total_down_values_num = []
    
    total_onepiece_keys_num = []
    total_onepiece_values_num = []
    
    total_outer_keys_num = []
    total_outer_values_num = []
    
    total_style_keys_num = []
    total_style_values_num = []
    
    save_inx = []
    save_image_name = []
    
    images = []
    
    for file in tqdm(label_path) :
    
        data = preprocessing_json(file)
        
        images.append(data)
    
    #with open("./images_attribute2.json", "w",encoding='UTF-8-sig') as json_file:
    #    json.dump(images, json_file,ensure_ascii=False)

    with open("images_attribute2.json", "r",encoding='UTF-8-sig') as st_json:
        file_x = json.load(st_json)
    
    images_df = pd.DataFrame(file_x) # moives 3883 개 #attribute 가진 수 만큼 만듬
    images_df = column_split(images_df)
    
    select_columns = [ 'check_up',
       'check_down', 'check_onepiece', 'check_outer', 'image_name',
       'json_index']
 
    images_df_meta = images_df.loc[:,select_columns]
    images_df_using = images_df.drop(columns=select_columns)
        
    b_use = images_df_meta['image_name'].apply(lambda x : index_check(x))
    images_df_using = images_df_using.loc[b_use,:] 
    images_df_meta = images_df_meta.loc[b_use,:] 
    images_df_using = images_df_using.reset_index(drop=True)
    images_df_meta = images_df_meta.reset_index(drop=True)
    
    images_df_meta['shop_name'] = images_df_meta['image_name'].str.split('_').str[0]
    images_df_meta['item_index'] = images_df_meta['image_name'].str.split('_').str[1]
    images_df_meta['item_order'] = images_df_meta['image_name'].str.split('_').str[2]
    images_df_meta['item_order'] = images_df_meta['item_order'].apply(lambda x : str(x).split('.')[0])

    search_idx = images_df_meta['item_order'] == '00'
    df_items = images_df_meta.loc[~search_idx,:]
    df_temps = df_items.loc[:,['shop_name','item_index']]
    df_temps = df_temps.drop_duplicates(['shop_name','item_index'],keep='first')
    
    images_df_using = images_df_using.iloc[df_temps.index,:] 
    images_df_meta = images_df_meta.iloc[df_temps.index,:] 
    
    images_df_using = images_df_using.reset_index(drop=True)
    images_df_meta = images_df_meta.reset_index(drop=True)

    # =============================================================================
    # 전처리된 df 저장 to dict
    # =============================================================================
    images_df_using.to_json('images_df_using_filter_format2.json',orient='table')
    images_df_meta.to_json('images_df_meta_filter_format2.json',orient='table')
 
