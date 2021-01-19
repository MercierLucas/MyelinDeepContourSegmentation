import matplotlib.pyplot as plt
import numpy as np
from image_processing import normalize

def compare_images(images,titles,img_to_show=2,add_heatmap=False):
    rows = len(images)
    for i in range(img_to_show):
        for r in range(0,rows):
            #img = images[r] if len(images[r][i].shape) <= 2 else images[r][i]

            img = images[r][i]
            if img.shape[-1] == 1:
                img = img.squeeze()
            

            plt.subplot(img_to_show,rows,i*rows+r+1)
            plt.title(titles[r])
            plt.imshow(img)
            if add_heatmap:
                plt.colorbar()
    plt.show()
