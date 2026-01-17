#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_folder = BASE_DIR

classes=['Canon','Epson','Hp']

data=[]

for cls in classes:
    folder=os.path.join(base_folder,cls)
    
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        
        data.append([img_path,cls.lower()])
        
df=pd.DataFrame(data,columns=['Image','Brand'])
    


# In[3]:


df_suffle=df.sample(frac=1).reset_index(drop=True)
df_suffle.head(10)


# In[4]:


from PIL import Image

def image_analyse(path):
    img=Image.open(path)
    print("Image mode (Channel Code)",img.mode)
    print("Image_size(Pixels)",img.size)
    print("DPI", img.info.get("dpi"))

for p in df['Image'].head(1):
    image_analyse(p)




# In[6]:


import cv2
import numpy as np
def preprocessing_size(img,size=(224,224)):
    return cv2.resize(img,size)

def gray_scale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
def normalize(img):
    return img/255.0


def load_tiff(path):
    return np.array(Image.open(path))
    


# In[7]:


def preprocessing(path):
    img=load_tiff(path)
    img=gray_scale(img)
    img=preprocessing_size(img,(224,224))
    img=normalize(img)
    img=img.reshape(224,224,1)
    return img
