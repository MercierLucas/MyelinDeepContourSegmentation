import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py
def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def plot_training_metrics(scores,val_scores,score_name):
    plt.plot(scores)
    plt.plot(val_scores)
    plt.title(f"Model {score_name} score")
    plt.ylabel(f"{score_name}")
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def compute_results(metrics_func,metrics_name,layers,y_true,y_pred):
    y_true = y_true.astype("float64")
    y_pred = y_pred.astype("float64")

    if len(metrics_func) != len(metrics_name):
        print("Warning, must have same number of names and func")
        return
    
    # Table's header
    header = f"{'':10} |"
    for i in layers:
        header += f"{i:10} |"
    print(header)

    for metric_id in range(len(metrics_func)):
        row = f"{metrics_name[metric_id]:10} |"
        for layer_idx in range(len(layers)):
            metric = metrics_func[metric_id](y_true[:,:,:,layer_idx],y_pred[:,:,:,layer_idx])
            row += f"{metric:10.3f} |"
        print(row)
