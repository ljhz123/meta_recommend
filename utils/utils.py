# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:21:33 2020

@author: Lee jung ho
"""
import matplotlib.pyplot as plt 

#plt.rcParams["figure.figsize"] = (800,600)

import os, shutil

import shutil
from tqdm import tqdm
import numpy as np
#data_path = os.path.join('./K-fashion(2)/K-fashion(2)')

import json
from glob import glob
from numpy import dot
from numpy.linalg import norm
import pandas as pd

import numpy as np

import matplotlib.image as mpimg
from PIL import Image
import cv2
import json
import itertools
import collections
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import numpy as np

def plot_2d_graph(vocabs, xs, ys,name='d2v',save=False):
    plt.figure(figsize=(8 ,6))
    plt.scatter(xs, ys, marker = 'o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
    if save :
        plt.savefig('./save_fig/{}_plot.png'.format(name))
        
def column_one_hot(df) :
    col = df.columns
    for i,col_idx in enumerate(col) : 
        df_dummy = pd.get_dummies(df[col[i]], prefix=col[i])
        df = pd.concat([df,df_dummy],axis=1)
        del df[col[i]]
    return df

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

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
        


def check_zero(li) :
    #li = [0,6,0,0,2]
    return np.sum([i !=0 for i in li])


def display_image(df,index) :
    #plt.style.use("default")
    jpg_img_arr = mpimg.imread('./datasets/{}/{}'.format(df['json_index'][index],df['image_name'][index]))
    
    return jpg_img_arr
    #plt.imshow(jpg_img_arr)
    #plt.xlabel(images_df['image_name'][index])
    #jpg_IMG = Image.open('./datasets/{}/{}'.format(tt['json_index'][0],tt['image_name'][0]))
    #jpg_IMG.show()

def search_best_neighbor_gt(images_df_meta,images_df_using,using_att_idx,query_idx,gt_pair,u_model,save=False) :
    
    fig = plt.figure()
    
    rows = 1
    cols = 2
    #query_idx=3
    
    best_item_index = int(gt_pair[query_idx])
    img1 = display_image(images_df_meta,query_idx)
    img2 = display_image(images_df_meta,best_item_index)
    
    query_att = using_att_idx[query_idx]
    rec_att = using_att_idx[best_item_index]
    
    result = [x for x in query_att if x in rec_att]
    
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img1)
    
    category_query = images_df_meta.iloc[query_idx,0:4]
    item_combine_query = category_query.index[category_query]
    item_combine_query = [kk.split('_')[1] for kk in item_combine_query] 
    
    category_rec = images_df_meta.iloc[best_item_index,0:4]
    item_combine_rec = category_query.index[category_rec]
    item_combine_rec = [kk.split('_')[1] for kk in item_combine_rec] 

    # ax1.set_title('query : {} \n style : {}/{}'.format(images_df_meta['image_name'][query_idx],
    #                                   images_df_using['스타일'][query_idx],
    #                                   images_df_using['서브스타일'][query_idx]))
    ax1.set_title('query : {} \n style : {}/{}'.format(item_combine_query,
                                                         images_df_using['스타일'][query_idx],
                                                         images_df_using['서브스타일'][query_idx]),fontsize=8)
    ax1.axis("off")
     
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(img2)
    
    ax2.set_title('rec : {} \n style : {}/{}'.format(item_combine_rec,
                                                       images_df_using['스타일'][best_item_index],
                                                       images_df_using['서브스타일'][best_item_index]),fontsize=8)
  
    # ax2.set_title('rec : {} \n style : {}/{}'.format(images_df_meta['image_name'][best_item_index],
    #                                   images_df_using['스타일'][best_item_index],
    #                                   images_df_using['서브스타일'][best_item_index]))
    
    ax2.axis("off")
    fig.suptitle('공통된 속성 : {} \n query 속성 : {} \n  rec 속성 : {}  '.\
                 format(result,query_att,rec_att),fontsize=6)
    if save :
        print('save figure')
        os.makedirs('./save_fig/{}'.format(u_model),exist_ok=True)
        plt.savefig('./save_fig/{}/query_{}_rec_{}.png'.format(u_model,query_idx,best_item_index))
    plt.show()



def column_split(images_df) :
    
    col = images_df.columns
    for i,col_idx in enumerate(col) : 
        #i=3
        first_idx = images_df.loc[images_df[col[i]] !='None',col[i]].index[0]
        if type(images_df.loc[images_df[col[i]] !='None',col[i]][first_idx] ) == list :
            try :
                images_df.loc[images_df[col[i]] == 'None',col[i]]  = ""
                
                tt_list = images_df[col[i]].tolist()
                temp = [0 if t == "" else len(t) for t in tt_list]
                #np.where(np.array(temp)==6)
                temp_max = np.max(temp)
               
                columns=['{}_'.format(col[i])+str(j+1) for j in range(temp_max)]
                #df3 = pd.DataFrame(images_df[col[i]].to_list(), columns=columns)
                temp_list = images_df[col[i]].tolist()
                temp_list = [tuple(li) for li in temp_list]
                images_df[columns] = pd.DataFrame(temp_list, 
                                   index= images_df.index)
                del images_df[col[i]]                
            except ValueError: 
                print(col[i]+' has problem')
                continue
                
    return images_df


def search_best_neighbor(images_df_meta,images_df_using,query_idx,top_score,top_neighbor_idx,u_model,save=False) :
    
    fig = plt.figure()
    rows = 1
    cols = 2
    #query_idx=3
    best_score_position = np.array(top_score[query_idx]).argmax()
    best_item_index = np.array(top_neighbor_idx)[query_idx,best_score_position]
    
    img1 = display_image(images_df_meta,query_idx)
    img2 = display_image(images_df_meta,best_item_index)
     
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img1)
    ax1.set_title('query : {} \n style : {}/{}'.format(images_df_meta['image_name'][query_idx],
                                      images_df_using['스타일'][query_idx],
                                      images_df_using['서브스타일'][query_idx]))
    ax1.axis("off")
     
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(img2)
    ax2.set_title('rec : {} \n style : {}/{}'.format(images_df_meta['image_name'][best_item_index],
                                      images_df_using['스타일'][best_item_index],
                                      images_df_using['서브스타일'][best_item_index]))
    ax2.axis("off")
    
    if save :
        print('save figure')
        os.makedirs('./save_fig/{}'.format(u_model),exist_ok=True)
        plt.savefig('./save_fig/{}/query_{}_rec_{}.png'.format(u_model,query_idx,best_item_index))
    plt.show()

def search_best_neighbor_gt_topk(images_df_meta,images_df_using,using_att_idx,query_idx,gt_pair,u_model,save=False,topn=20) :
    
    for top_idx in range(topn):
        try :   
   
             
            fig = plt.figure()
            rows = 1
            cols = 2
            
            #images_df_meta = images_df_meta
            #images_df_using = images_df_using
            #using_att_idx = using_att_idx
            #query_idx='0'
            #gt_pair= gt_pair
            
            #if type(query_idx) == int
            best_item_index = int(gt_pair[query_idx][top_idx])
            img1 = display_image(images_df_meta,query_idx)
            img2 = display_image(images_df_meta,best_item_index)
            
            query_att = using_att_idx[query_idx]
            rec_att = using_att_idx[best_item_index]
            
            result = [x for x in query_att if x in rec_att]
            
            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(img1)
            
            category_query = images_df_meta.iloc[query_idx,0:4]
            item_combine_query = category_query.index[category_query]
            item_combine_query = [kk.split('_')[1] for kk in item_combine_query] 
            
            category_rec = images_df_meta.iloc[best_item_index,0:4]
            item_combine_rec = category_query.index[category_rec]
            item_combine_rec = [kk.split('_')[1] for kk in item_combine_rec] 
        
            # ax1.set_title('query : {} \n style : {}/{}'.format(images_df_meta['image_name'][query_idx],
            #                                   images_df_using['스타일'][query_idx],
            #                                   images_df_using['서브스타일'][query_idx]))
            ax1.set_title('image : {} \n query : {} \n style : {}/{}'.format(images_df_meta['image_name'][query_idx],
                                                               item_combine_query,
                                                               images_df_using['스타일'][query_idx],
                                                               images_df_using['서브스타일'][query_idx]),fontsize=8)
            ax1.axis("off")
             
            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(img2)
            
            ax2.set_title('image : {} \n rec : {} \n style : {}/{}'.format(images_df_meta['image_name'][best_item_index],
                                                                       item_combine_rec,
                                                                       images_df_using['스타일'][best_item_index],
                                                               images_df_using['서브스타일'][best_item_index]),fontsize=8)
          
            # ax2.set_title('rec : {} \n style : {}/{}'.format(images_df_meta['image_name'][best_item_index],
            #                                   images_df_using['스타일'][best_item_index],
            #                                   images_df_using['서브스타일'][best_item_index]))
            
            ax2.axis("off")
            fig.suptitle('공통된 속성 : {} \n query 속성 : {} \n  rec 속성 : {}  '.\
                         format(result,query_att,rec_att),fontsize=8)
            if save :
                print('save figure')
                os.makedirs('./save_fig/{}/{}'.format(u_model,query_idx),exist_ok=True)
                plt.savefig('./save_fig/{}/{}/top{}_rec_{}.png'.format(u_model,query_idx,top_idx,best_item_index))
                #plt.show()
            #os.makedirs('./save_fig/{}/{}'.format(u_model,query_idx),exist_ok=True)
            
            # with open("./save_fig/{}/{}/rec_items.txt".format(u_model,query_idx), "w") as file:
            #         file.write("rec item list \n")
            
            with open("./save_fig/{}/{}/rec_items.txt".format(u_model,query_idx), "a") as file:
                file.write("{} \n".format(images_df_meta['image_name'][best_item_index]))
                file.close()
            
        except :
            continue

  
 
def preprocessing_json(json_file_path,Train=True) :
    
    up_attribte = ['소재','카테고리','옷깃','디테일','기장','핏','넥라인','프린트']
    down_attribte = ['소재','카테고리','디테일','기장','핏','프린트']
    onpiece_attribte = ['소재','카테고리','옷깃','디테일','기장','핏','넥라인','프린트']
    outer_attribte = ['소재','카테고리','옷깃','디테일','기장','핏','넥라인','프린트']
    style_attribte = ['스타일','서브스타일']
    check_attribte = ['check_up','check_down','check_onepiece','check_outer']     
    
    try : 
        #file = label_path[0]
        with open(json_file_path, "r", encoding="utf8") as st_json:
            st_python = json.load(st_json)
    except  :
        print('json file is error')
        
        
    data = {}
    
    up = st_python['데이터셋 정보']['데이터셋 상세설명']['라벨링']['상의'][0]
    keys = list(up.keys())    
    for key in up_attribte : 
        k = '상의_'+key
        try :
            data[k] = up[key]
        except KeyError :
            data[k] = 'None'
    
    down = st_python['데이터셋 정보']['데이터셋 상세설명']['라벨링']['하의'][0]
    keys = list(down.keys())    
    for key in down_attribte : 
        k = '하의_'+key
        try :
            data[k] = down[key]
        except KeyError :
            data[k] = 'None'
    
    onepiece = st_python['데이터셋 정보']['데이터셋 상세설명']['라벨링']['원피스'][0]
    for key in onpiece_attribte : 
        k = '원피스_'+key
        try :
            data[k] = onepiece[key]
        except KeyError :
            data[k] = 'None'
            
    outer = st_python['데이터셋 정보']['데이터셋 상세설명']['라벨링']['아우터'][0]
    for key in outer_attribte : 
        k = '아우터_'+key
        try :
            data[k] = outer[key]
        except KeyError :
            data[k] = 'None'
  
    style = st_python['데이터셋 정보']['데이터셋 상세설명']['라벨링']['스타일'][0]
    for key in style_attribte : 
       k = key
       try :
           data[k] = style[key]
       except KeyError :
           data[k] = 'None'
    check = [len(up),len(down),len(onepiece),len(outer)]
    check = [True if c != 0 else False for c in check]
    
    for i,key in enumerate(check_attribte) :
        data[key] = check[i]
        
    if Train :     
        data['image_name'] = st_python['이미지 정보']['이미지 파일명']
        data['json_index'] = json_file_path.split('\\')[1]
    
    # attribute_list = ['니트웨어','노멀','타이트','브이넥','니트','단추','무지']
    return data


