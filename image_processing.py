import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
from PIL import Image
import cv2


def load_specific_images(path,images,shape=None,dilatation=None):
  base_type=".tif"
  mask_type=".png"

  X = []
  Y_cell = []
  Y_border = []

  for image in images:
    filename = image.replace(base_type,"")
    #img = Image.open(path+filename+base_type)
    img = cv2.imread(path+filename+base_type,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img_to_array(img)
    #mask = Image.open(path+filename+"_segmented"+mask_type).convert("RGB")
    
    mask_cell = cv2.imread(path+filename+"_segmented"+mask_type)
    borders = cv2.imread(path+filename+"_segmented_EXT"+mask_type,1)
    
    # keeping only borders
    if dilatation != None:
      if len(dilatation) != 3:
        print("Warning, dilatation must be dilatation=(x,y,iterations)")
      mask_border = np.zeros((borders.shape[0],borders.shape[1],2))
      mask_border[:,:,0] = (borders[:,:,1] < 120).astype("float32")
      kernel = np.ones((dilatation[0],dilatation[1]),np.uint8)
      mask_border[:,:,0] = cv2.dilate(mask_border[:,:,0],kernel,iterations = dilatation[2])
      mask_border[:,:,0] = (mask_border[:,:,0] > 0.5).astype("float32")
      mask_border[:,:,1] = 1 - mask_border[:,:,0]
    #mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if shape != None:
        if len(shape) == 2:
            img = cv2.resize(img,(shape[0],shape[1]))
            mask_cell = cv2.resize(mask_cell,(shape[0],shape[1]))
            mask_border = cv2.resize(mask_border,(shape[0],shape[1]))

        else:
            print("Please provide shape as (X,Y)")

    X.append(img)
    Y_cell.append(mask_cell)
    Y_border.append(mask_border)
    
  return np.array(X),np.array(Y_cell),np.array(Y_border)



def load_full_images(path,shape=None,base_type=".tif",mask_type=".png",dilatation=None,escape=[]):
    X = []
    Y_cell = []
    Y_border = []

    ids = next(os.walk(path))[2]
    for i in range(len(ids)):

        if ("_Contours" not in ids[i]) and ("segmented" not in ids[i]):
            filename = ids[i].replace(base_type,"")

            if filename in escape:
              print(f"Escaped {ids[i]}")
              continue

            #img = Image.open(path+filename+base_type)
            img = cv2.imread(path+filename+base_type,0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = img_to_array(img)
            #mask = Image.open(path+filename+"_segmented"+mask_type).convert("RGB")
            
            mask_cell = cv2.imread(path+filename+"_segmented"+mask_type)
            borders = cv2.imread(path+filename+"_segmented_EXT"+mask_type,1)
            
            # keeping only borders
            if dilatation != None:
              if len(dilatation) != 3:
                print("Warning, dilatation must be dilatation=(x,y,iterations)")
              mask_border = np.zeros((borders.shape[0],borders.shape[1],2))
              mask_border[:,:,0] = (borders[:,:,1] < 120).astype("float32")
              kernel = np.ones((dilatation[0],dilatation[1]),np.uint8)
              mask_border[:,:,0] = cv2.dilate(mask_border[:,:,0],kernel,iterations = dilatation[2])
              mask_border[:,:,0] = (mask_border[:,:,0] > 0.5).astype("float32")
              mask_border[:,:,1] = 1 - mask_border[:,:,0]
            #mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if shape != None:
                if len(shape) == 2:
                    img = cv2.resize(img,(shape[0],shape[1]))
                    mask_cell = cv2.resize(mask_cell,(shape[0],shape[1]))
                    mask_border = cv2.resize(mask_border,(shape[0],shape[1]))

                else:
                    print("Please provide shape as (X,Y)")

            X.append(img)
            Y_cell.append(mask_cell)
            Y_border.append(mask_border)

                
    return np.array(X),np.array(Y_cell),np.array(Y_border)


def apply_dilatation(x,thickness=(5,5),iterations=1):
    res = np.zeros((x.shape)).astype("float32")
    for img_idx in range(x.shape[0]):
        res[img_idx,:,:] = cv2.dilate(x[img_idx,:,:],thickness,iterations = iterations)
    return res

  
def normalize(x,norm=255,data_type="float64"):
  res = np.zeros(x.shape)
  for idx in range(x.shape[-1]):
    layer = x[:,:,:,idx]
    min_ = np.min(layer)
    max_ = np.max(layer)
    res[:,:,:,idx] = (layer - min_)/(max_-min_) * norm
    res[:,:,:,idx] = res[:,:,:,idx]
    
  return res.astype(data_type)

def removeMiddleLayer(masks):
  result = []
  w = masks.shape[1]
  h = masks.shape[2]
  for mask in masks:
    temp = np.empty((w,h,2))
    temp[:,:,0] = mask[:,:,0]
    temp[:,:,1] = mask[:,:,2]
    result.append(temp)

  return np.array(result)

def keep_borders_only(masks):
  res = []
  for mask in masks:
    res.append(get_borders(mask))
  return np.array(res)


def converToShape(X,shape):
  if shape == 2:
    return cv2.merge((X,X))
  elif shape == 3:
    return cv2.merge((X,X,X))

def raise_to_three_layers(X,fill_with=0):
    h = X.shape[1]
    w = X.shape[2]
    res = np.zeros((X.shape[0],h,w,3))
    res[:,:,:,0] = X[:,:,:,0]
    res[:,:,:,1] = X[:,:,:,1]
    if fill_with == 0:
      res[:,:,:,2] = np.zeros((h,w))
    else:
      res[:,:,:,2] = np.ones((h,w))
    return res