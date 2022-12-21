import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

CONFIG_FILE_NAME = 'config.json'

def load_config():
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)
    return config

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def imshow(img, ax=None):
    if len(img.shape)==3:
        img_for_show = np.transpose(img.numpy(), (1, 2, 0))
    else:
        img_for_show = img.numpy().T
    if ax:
        ax.imshow(img_for_show,cmap='gray')
        ax.axis("off")
    else:
        plt.imshow(img_for_show,cmap='gray')
        plt.axis("off")
        plt.show() 
       