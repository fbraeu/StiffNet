#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import imageio
import numpy as np
import pandas as pd
import shutil
import cv2
from matplotlib import pyplot as plt
import random


# In[23]:


home = os.path.expanduser("~")
csv_file = home+"/workspace/StiffnessClassification/Classification/Preprocessing/deformation_grading_thanadet.csv"
img_dir = home+"/workspace/StiffnessClassification/Classification/Preprocessing/data_raw/ONHD/"
seg_dir = home+"/workspace/StiffnessClassification/Classification/Preprocessing/data_raw/ONHD_seg/"
masked_dir = home+"/workspace/StiffnessClassification/Classification/Preprocessing/data_raw/ONHD_masked/"
save_dir = home+"/workspace/StiffnessClassification/Classification/Preprocessing/data_processed/"


# In[24]:


def GetFilesFromDir(dir):
    
    file_paths = []
    for root, dirs, files in os.walk(os.path.abspath(dir)):
        for file in files:
            file_paths.append(os.path.join(root, file))
    file_paths.sort()
    
    return file_paths


img_paths = GetFilesFromDir(img_dir)
seg_paths = GetFilesFromDir(seg_dir)


# In[25]:


def MaskIMG(img_paths, seg_paths, masked_dir):
    
    if not os.path.exists(masked_dir):
        os.makedirs(masked_dir)
    else:
        shutil.rmtree(masked_dir)
        os.makedirs(masked_dir)
    
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_paths[i], cv2.IMREAD_GRAYSCALE)
        seg[seg!=0] = 1
        
        filename = img_paths[i].split("/")[-1]
        plt.imsave(masked_dir+filename, np.multiply(img,seg), cmap="gray")
        
        
MaskIMG(img_paths, seg_paths, masked_dir)


# In[26]:


def GetLowHighDefID(csv_file):
    
    df = pd.read_csv(csv_file)
    lowdef_id_tmp = np.array((df.loc[df['Grade'] == 1]).ID)
    highdef_id_tmp = np.array((df.loc[df['Grade'] == 2]).ID)
    
    lowdef_id = [str(item).zfill(3) for item in lowdef_id_tmp]
    highdef_id = [str(item).zfill(3) for item in highdef_id_tmp]

    return lowdef_id, highdef_id


lowdef_id, highdef_id = GetLowHighDefID(csv_file)


# In[27]:


def GetLowHighDefIMG(lowdef_id, highdef_id, img_paths):
    
    lowdef_img_paths = []
    for i in range(len(lowdef_id)):
        for j in range(len(img_paths)):
            if img_paths[j].find(lowdef_id[i]) != -1:
                lowdef_img_paths.append(img_paths[j])
                break
    
    highdef_img_paths = []
    for i in range(len(highdef_id)):
        for j in range(len(img_paths)):
            if img_paths[j].find(highdef_id[i]) != -1:
                highdef_img_paths.append(img_paths[j])
                break
    
    assert len(lowdef_id)==len(lowdef_img_paths) and len(highdef_id)==len(highdef_img_paths), "Lists have to have the same length!"
    
    return lowdef_img_paths, highdef_img_paths


lowdef_img_paths, highdef_img_paths = GetLowHighDefIMG(lowdef_id, highdef_id, img_paths)


# In[28]:


def CopyLowHighDefIMG(lowdef_img_paths, highdef_img_paths, dest_dir, split=None):
    
    if split is not None:
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir+"Train/LowDef")
            os.makedirs(dest_dir+"Train/HighDef")
            os.makedirs(dest_dir+"Val/LowDef")
            os.makedirs(dest_dir+"Val/HighDef")
            os.makedirs(dest_dir+"Test/LowDef")
            os.makedirs(dest_dir+"Test/HighDef")
        else:
            shutil.rmtree(dest_dir)
            os.makedirs(dest_dir+"Train/LowDef")
            os.makedirs(dest_dir+"Train/HighDef")
            os.makedirs(dest_dir+"Val/LowDef")
            os.makedirs(dest_dir+"Val/HighDef")
            os.makedirs(dest_dir+"Test/LowDef")
            os.makedirs(dest_dir+"Test/HighDef")
        
        if len(lowdef_img_paths) < len(highdef_img_paths):
            # Low deformation images
            length_test_low = int(split[2]*len(lowdef_img_paths))
            length_val_low = int(split[1]*len(lowdef_img_paths))
            length_train_low = len(lowdef_img_paths) - length_val_low - length_test_low
            
            # High deformation images
            length_train_high = len(highdef_img_paths) - length_val_low - length_test_low
            length_val_high = length_val_low
            length_test_high = length_test_low
        else:
            # Low deformation images
            length_test_high = int(split[2]*len(highdef_img_paths))
            length_val_high = int(split[1]*len(highdef_img_paths))
            length_train_high = len(highdef_img_paths) - length_val_high - length_test_high
            
            # High deformation images
            length_train_low = len(lowdef_img_paths) - length_val_high - length_test_high
            length_val_low = length_val_high
            length_test_low = length_test_high
        
        
        print("HighDef: Train({}), Val({}), Test({}), LowDef: Train({}), Val({}), Test({})".format(length_train_high,
                                                                                                   length_val_high,
                                                                                                   length_test_high,
                                                                                                   length_train_low,
                                                                                                   length_val_low,
                                                                                                   length_test_low))
        
        
        # Low deformation images
        random.shuffle(lowdef_img_paths)

        for i in range(length_train_low):
            shutil.copy(lowdef_img_paths[i], dest_dir+"Train/LowDef")

        for i in range(length_train_low, (length_train_low+length_val_low)):    
            shutil.copy(lowdef_img_paths[i], dest_dir+"Val/LowDef")

        for i in range((length_train_low+length_val_low), len(lowdef_img_paths)):
            shutil.copy(lowdef_img_paths[i], dest_dir+"Test/LowDef")
            
        
        # High deformation images
        random.shuffle(highdef_img_paths)

        for i in range(length_train_high):
            shutil.copy(highdef_img_paths[i], dest_dir+"Train/HighDef")

        for i in range(length_train_high, (length_train_high+length_val_high)):    
            shutil.copy(highdef_img_paths[i], dest_dir+"Val/HighDef")

        for i in range((length_train_high+length_val_high), len(highdef_img_paths)):
            shutil.copy(highdef_img_paths[i], dest_dir+"Test/HighDef")

        
    else:
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir+"LowDef")
            os.makedirs(dest_dir+"HighDef")
        else:
            shutil.rmtree(dest_dir)
            os.makedirs(dest_dir+"LowDef")
            os.makedirs(dest_dir+"HighDef")

        for i in range(len(lowdef_img_paths)):
            shutil.copy(lowdef_img_paths[i], dest_dir+"LowDef")

        for i in range(len(highdef_img_paths)):
            shutil.copy(highdef_img_paths[i], dest_dir+"HighDef")
        
        
CopyLowHighDefIMG(lowdef_img_paths, highdef_img_paths, save_dir, split=(0.85,0.15,0.0))


# In[ ]:




