#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import shutil
import torch
import torch.nn.functional as F


# # Get all file paths in specified directory

# In[7]:


def GetFilesFromDir(dir):
    file_paths = []
    for root, dirs, files in os.walk(os.path.abspath(dir)):
        for file in files:
            file_paths.append(os.path.join(root, file))
    file_paths.sort()
    return file_paths


# # Save checkpoint (indicate best checkpoint)

# In[8]:


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


# # Computes and stores the average and current value

# In[9]:


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


# # Maintain size of image after convolution block
# Pads image with zeros to convert size if image to target size

# In[36]:


def make_same_size(in_img, target_shape):
    in_height, in_width = in_img.shape[2], in_img.shape[3]
    target_height, target_width = target_shape[0], target_shape[1]

    #Finally, the padding on the top, bottom, left and right are:
    if in_height == target_height:
        pad_top, pad_bot = 0, 0
    else:
        pad_top = (target_height - in_height) // 2
        pad_bot = (target_height - in_height) - pad_top
    if in_width == target_width:
        pad_left, pad_right = 0, 0
    else:
        pad_left = (target_width - in_width) // 2
        pad_right = (target_width - in_width) - pad_left   

    return F.pad(in_img, (pad_left, pad_right, pad_top, pad_bot), "constant", 0)


# In[ ]:





# In[ ]:




