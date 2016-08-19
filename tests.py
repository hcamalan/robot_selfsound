# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:43:55 2016

@author: huseyin
"""

import numpy as np
import functions as f
import matplotlib.pyplot as plt


def plot_components(mat):
    plt.figure(figsize = (16,12))
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.imshow(mat[:, i, :], aspect = "auto", interpolation = "none")
        plt.clim(vmax = np.max(mat), vmin=np.min(mat))
        plt.colorbar()
    plt.show()
    return 0

def flatten_recreate_test(mat):
    """
    Checks if a matrix matches its original version after being flattened and refolded on an axis.
    """
    temp = f.recreate_ax(f.flatten_ax(mat), mat.shape[-1])
    return np.all(mat == temp)   
    
def rescale_max_check_test(mat):
    """
    Checks if a matrix has only ones as maxes after being rescaled twice.
    """
    maxes1, rescaled1 = f.rescale(mat)
    maxes2, rescaled2 = f.rescale(rescaled1)
    return np.all(maxes2 == np.ones(maxes2.shape))
    
def rescale_max_check_test_2(mat):
    """
    Similar to before, checks that maximum value of a matrix after rescaling is exactly 1.
    """
    maxes, rescaled = f.rescale(mat)
    return np.max(np.abs(rescaled)) == 1.
    
def rescale_data_check_test(mat):
    """
    Checks if a matrix matches its original version after being rescaled and re-multiplied
    with its maximum values.
    
    Be aware that numerical errors can cause the test to not pass, which is why the 
    difference is rounded.
    """
    maxes, rescaled = f.rescale(mat)
    orig = f.reverse_rescale(maxes, rescaled)
    diff = np.round(mat - orig, 10)    
    return np.all(diff == 0.)
    
def convert_to_z_test(mat):
    """
    Checks if a matrix matches its original version after being converted to and reconverted 
    from z-values. 
    """
    means, stds, z_mat = f.convert_to_z(mat)
    reconv = f.unconvert_from_z(means, stds, z_mat)
    return np.all(np.round(reconv-mat)==0)

def convert_to_z_test_2(mat, ind = 0):
    """
    Similar to convert_to_z_test, but removes f.unconvert_from_z from the process and 
    checks just one index 
    """
    means, stds, z_mat = f.convert_to_z(mat)
    return np.all((z_mat[:, ind, :] * stds[ind]) + means[ind] == mat[:, ind, :])
    
def convert_to_z_test_3(mat):
    """
    Converting to z values truly rescales the series, in that their sum should become zero.
    This function checks whether this property holds for a given matrix.
    """
    means, stds, z_mat = f.convert_to_z(mat)
    flattened = f.flatten_ax(z_mat)
    return np.all(np.round(np.sum(flattened, axis = 0)) == 0)
    
def preprocessing_test(mat):
    """
    Checks if a matrix matches its original version after being preprocessed and reverse_preprocessed.
    """
    means, stds, maxes, temp = f.preprocess(mat)
    temp2 = f.reverse_preprocess(means, stds, maxes, temp)
    diff = np.round(mat - temp2, 10)
    return np.all(diff == 0.)

def make_chunks_test(mat, chunks):
    """
    Separates data into smaller chunks and checks if the smaller chunks combine to 
    recreate the intact dataset.
    """
    temp = np.zeros(mat.shape)
    sh = chunks.shape
    for i in range(sh[-1]):
        temp[sh[0]*i:sh[0]*(i+1), :, :] = chunks[:, :, :, i]
    return np.all(temp == mat)


def confusion_matrix_test(n_cls = 3):
    dummy_dat = np.ones((100, n_cls))
    for i in range(0, n_cls):
        dummy_dat[:, i] *= i
    return np.all(np.eye(n_cls)==f.calculate_confusion_matrix(dummy_dat, np.arange(n_cls), n_nets = n_cls))
#    
def learn_predict_test():
    length = 1000
    mfcc = np.zeros((length, 1))
    j_input = np.vstack((np.linspace(0, 10, length), np.linspace(0, 1, length))).T
    mfcc[:, 0] = 10*j_input[:, 1] - 2*(np.power(j_input[:, 0], 0.5))
    mfcc, j_input = mfcc/np.max(mfcc), j_input/np.max(j_input)
    preds = f.predict_mfcc(f.learn_mfcc(mfcc, j_input, epochs = 50, inter_layer = 10), j_input, n_mfcc = 1)
#    plt.plot(preds)
#    plt.plot(mfcc)
#    plt.show()
    return np.sum(np.abs(preds-mfcc))/np.sum(np.abs(mfcc))

