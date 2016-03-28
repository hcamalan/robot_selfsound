# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:31:15 2016

@author: huseyin

List of functions:

prep_data
get_mfcc
get_position
get_joint_names
sync_walks
sync_input_cc
convert_to_z
flatten_ax
recreate_ax
normalize
learn_mfcc
predict_mfcc
calculate_confusion_matrix

"""

import numpy as np
import matplotlib.pyplot as plt
from pybrain import datasets
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

    
    

def prep_data(kwargs):
    """
    dict -> (np.array, np.array, np.array, np.array)
    
    Prepares the data from the "mfccJoints_*" files using the parameters provided by the user.
    In order, objects returned are mfcc, initial joint input, joint position change, and class.
    
    names (list): list including filenames. For example, to load mfccJoints_carpet1.txt, just provide "carpet1".
    classes(list): list including classes. This list should match that of "names" both in length and order.
    n_joints(int): Number of joints provided in the robot data. This is 25 in the Aldebaran Nao recordings.
    t_cutoff(list(int) of 2): This will crop the relevant part of the encoding.
    endpoint(int): Kind of useless parameter that should be replaced by t_cutoff[1].
                   This was created because not all mfccJoints_ files were the same size.
    dim(int): Length of the 2nd dimension
    
    """
    n_joints = kwargs['n_joints']
    names = kwargs['names']
    t_cutoff = kwargs['t_cutoff']    
    endpoint = kwargs['endpoint']
    dim = kwargs['dim']
    classes = kwargs['classes']    
    
    mfccJoints = np.empty((t_cutoff[1]-t_cutoff[0] + 1, dim, len(names)))
    
    for ind, n  in enumerate(names):
        name = "mfccJoints_" + n + ".txt"
        temp = np.genfromtxt(name, delimiter = " ")[:t_cutoff[1] + 1, :]
        mfccJoints[:, :, ind] = temp[t_cutoff[0]:,:]
        #print(name)
    
    #Initialize matrices for joint positions and position change
    #We will throw away the +1 datapoint later
    j_0 = np.zeros((t_cutoff[1] - t_cutoff[0] + 1, n_joints, len(names)))
    j_delta = j_0.copy()

    #Insert information for each separate joint
    for sample in range(0, len(names)):
        for joint in range(0,n_joints):
            #Temporary variables
            pos_ind_j = get_position(joint, mfccJoints[:, :, sample])
            
            j_0[:, joint, sample] = pos_ind_j[:, 0]
            j_delta[:, joint, sample] = pos_ind_j[:, 0] - pos_ind_j[:, 1]
    
    #Roll joint input backwards by 1 and throw away the extra datapoint (the first one)
    j_delta, j_0 = np.roll(j_delta, -1, axis = 0)[:-1, :, :], np.roll(j_0, -1, axis = 0)[:-1, :, :]
    
    #Get all mfccs
    #Throw away the first datapoint because it's useless
    mfcc = get_mfcc(mfccJoints)[1:, :]
    return mfcc, j_0, j_delta



def get_mfcc(data):
    """
    np.array -> np.array
    
    Extracts the 13 mfcc components from the large dataset.
    """
    return data[:, 3:16]
    
    

def get_position(joint, data):
    """
    (int, np.array) -> np.array
    
    Extracts the initial and final positions of the joints, in that order.
    """
    ind_joint = joint+17
    return np.vstack((data[:, ind_joint], data[:, ind_joint + 76])).T



def get_joint_names():
    """
    () -> list
    
    Gets joint names in a list. Kind of unnecessary, but improves readability.
    """
    joints = "HEAD_PITCH HeadPitch HEAD_YAW HeadYaw L_ANKLE_PITCH LAnklePitch L_ANKLE_ROLL LAnkleRoll L_ELBOW_ROLL LElbowRoll L_ELBOW_YAW LElbowYaw L_HAND LHand L_HIP_PITCH LHipPitch L_HIP_ROLL LHipRoll L_HIP_YAW_PITCH LHipYawPitch L_KNEE_PITCH LKneePitch L_SHOULDER_PITCH LShoulderPitch L_SHOULDER_ROLL LShoulderRoll L_WRIST_YAW LWristYaw R_ANKLE_PITCH RAnklePitch R_ANKLE_ROLL RAnkleRoll R_ELBOW_ROLL RElbowRoll R_ELBOW_YAW RElbowYaw R_HAND RHand R_HIP_PITCH RHipPitch R_HIP_ROLL RHipRoll R_KNEE_PITCH RKneePitch R_SHOULDER_PITCH RShoulderPitch R_SHOULDER_ROLL RShoulderRoll R_WRIST_YAW RWristYaw"
    joints = joints.split(" ")
    joints = joints[1::2]
    return joints



def sync_walks(mfcc, sync_period = [0, -1], reference = 0, component = 0):
    """
    (np.array, [list of int], [int], [int]) -> np.array
    Takes as input the combined mfcc from multiple walks and removes the phase difference.
    Should be used before the sync_input_cc function.
    """
    shifted_mfcc = mfcc.copy()*0.
    n_walks = mfcc.shape[2]
    cycle = 40
    #inter_mfcc_phase = np.zeros(n_walks)
    
    for walk in range(0, n_walks):
        #Optimal size for one cycle
        err = np.zeros(cycle)
        ind_ = np.arange(-cycle/2, cycle/2)
        
        for i1, i2 in enumerate(ind_):
            #mfcc[:, 0, 0] is reference, we compare everything else
            err[i1] = np.sum(np.abs(mfcc[sync_period[0]:sync_period[1], component, reference] -
                      np.roll(mfcc[sync_period[0]:sync_period[1], component, walk], i2)))
        
        phase = int(ind_[np.argmin(err)])
        #print phase
        shifted_mfcc[:, :, walk] = np.roll(mfcc[:, :, walk], phase, axis = 0)

    return shifted_mfcc


def sync_input_cc(mfcc, j_input, incl_joints = [2, 3, 8, 9, 10, 14, 15, 20, 21]):
    """
    (np.array, np.array, [list of int]) -> np.array
    
    Removes the phase shift artifacts between the joint inputs and the mfcc, using crosscorrelation. 
    """
    sync_signal = j_input[:, incl_joints, :]
    shifted_j = j_input.copy()*0.
    n_walks = mfcc.shape[2]
    hips = np.sum(sync_signal, axis = 1)
    hips = hips/np.max(hips, axis=0)

    for walk in range(0, n_walks):
        corr = np.correlate(hips[:, walk], mfcc[:, 0, walk], "same")
        phase = len(corr)/2 - np.argmax(corr)
        sh = np.roll(hips[:, walk], phase)
        shifted_j[:, :, walk] = np.roll(j_input[:, :, walk], phase, axis = 0)

    return shifted_j


def convert_to_z(mat):
    """
    (np.array) -> np.array
    Returns z values of an array with dimensions 2,3
    """
    if len(mat.shape) == 2:
        means, stds = np.tile(np.average(mat, axis = 0), (mat.shape[0], 1)), np.tile(np.std(mat, axis = 0), (mat.shape[0], 1))
        mat = (mat - means) / stds
    if len(mat.shape) > 2:
        flat = flatten_ax(mat)
        means, stds = np.tile(np.average(flat, axis = 0), (flat.shape[0], 1)), np.tile(np.std(flat, axis = 0), (flat.shape[0], 1))
        flat = (flat - means) / stds
        mat = recreate_ax(flat, mat.shape[-1])
    
    return means[0,:], stds[0, :], mat



#flatten_ax and recreate_ax are supplement functions for convert_to_z
def flatten_ax(mat):
    """
    (np.array) -> np.array
    Flattens 3-dim to 2-dim
    Assistance function to convert_to_z
    """
    a = mat.shape
    b = np.empty((a[0]*a[2], a[1]))
    
    for i in range(a[2]):
        b[i*a[0]:(i+1)*a[0], :] = mat[:, :, i]
    return b



def recreate_ax(mat, dim):
    """
    (np.array) -> np.array
    Reshapes 2-dim to 3-dim
    Assistance function to convert_to_z
    """
    a = mat.shape
    hi_dim = np.empty((a[0]/dim, a[1], dim))
    for i in range(dim):
        #print i*a[0], (i+1)*a[0]
        hi_dim[:, :, i] = mat[i*a[0]/dim:(i+1)*a[0]/dim, :]
    return hi_dim 


def normalize(data):
    """
    (np.array) -> (np.array)
    Normalizes data to prepare them for use by neural networks
    """
    dims = data.shape
    for i in range(0, dims[1]):
        for j in range(0, dims[2]):
            data[:, i, j] /= np.max(data[:, i, j])
    return data





def learn_mfcc(mfcc, j_input, epochs = 100, inter_layer = 20):
    """
    (np.array, np.array, [int], [int]) -> pybrain object
     
    Takes mfcc and joint input and returns a neural net trained using backpropagation.
    mfcc: mfcc object
    j_input: joint input, length of 0th dimension must be same as mfcc
    epochs: # of iterations (default 100)
    inter_layer: number of hidden layers (default 20)
    """
    print "Learning"
    n_mfcc = mfcc.shape[1]
    #j_input = np.concatenate((j_delta, j_0), axis = 1)
    dims = np.shape(j_input)    
    
    j_input = j_input/np.max(j_input)
    mfcc = mfcc/np.max(mfcc)
    tr_ds = datasets.SupervisedDataSet(dims[1], n_mfcc)
    
    for i in range(0, dims[0]):
        tr_ds.addSample(j_input[i, :], mfcc[i, :])

    #net = buildNetwork(1, 1, 1, bias=True)
    net = buildNetwork(dims[1], inter_layer, n_mfcc, bias=True)
    trainer = BackpropTrainer(net, tr_ds)
    #error = np.empty(epochs)    
    
    for epo in range(0, epochs):
        
        print trainer.train()
        #trainer.train()
    return net

def predict_mfcc(net, j_input, n_mfcc = 13):
    """
    (pybrain object, np.array, int) -> np.array
    Predicts output using an input and a net trained by the function 'learn_mfcc'
    net: net object from pybrain, obtained through learn_mfcc
    j_input: joint input that will be used to predict output
    n_mfcc: # of mfcc components (default 13)
    """
    dims = np.shape(j_input)
    
    mfcc_pred  = np.empty((dims[0], n_mfcc))

    
    for i in range(0, dims[0]):
        mfcc_pred[i, :] = net.activate(j_input[i, :])

    return mfcc_pred


def calculate_confusion_matrix(preds_aw, cls, n_nets = 3):
    """
    (np.array, np.array, [int]) -> np.array
    Calculates the confusion matrix given the predictions for all walks and the classes of these walks.
    """
    conf_mat = np.zeros((n_nets, n_nets))
    for i in range(0, n_nets):
        preds_reshaped = np.reshape(preds_aw[:, cls == i], -1)
        for j in range(0, n_nets):
            conf_mat[j, i] = np.round(np.sum(preds_reshaped == j)/float(len(preds_reshaped)), decimals = 2)
    return conf_mat


