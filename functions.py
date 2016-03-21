# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:47:02 2015

@author: huseyin
"""

import numpy as np
import matplotlib.pyplot as plt
from pybrain import datasets
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def get_joint_names():
    joints = "HEAD_PITCH HeadPitch HEAD_YAW HeadYaw L_ANKLE_PITCH LAnklePitch L_ANKLE_ROLL LAnkleRoll L_ELBOW_ROLL LElbowRoll L_ELBOW_YAW LElbowYaw L_HAND LHand L_HIP_PITCH LHipPitch L_HIP_ROLL LHipRoll L_HIP_YAW_PITCH LHipYawPitch L_KNEE_PITCH LKneePitch L_SHOULDER_PITCH LShoulderPitch L_SHOULDER_ROLL LShoulderRoll L_WRIST_YAW LWristYaw R_ANKLE_PITCH RAnklePitch R_ANKLE_ROLL RAnkleRoll R_ELBOW_ROLL RElbowRoll R_ELBOW_YAW RElbowYaw R_HAND RHand R_HIP_PITCH RHipPitch R_HIP_ROLL RHipRoll R_KNEE_PITCH RKneePitch R_SHOULDER_PITCH RShoulderPitch R_SHOULDER_ROLL RShoulderRoll R_WRIST_YAW RWristYaw"
    joints = joints.split(" ")
    joints = joints[1::2]
    return joints

#Define functions to get data
def get_mfcc(data):
    return data[:, 3:16]

def get_position(joint, data):
    """
    Initial position, final position
    """
    ind_joint = joint+17
    #Returns the initial and the final positions of a specific joint based on the 
    return np.vstack((data[:, ind_joint], data[:, ind_joint + 76])).T

def get_motor(joint, data):
    """
    Initial command, final command
    """
    ind_joint = joint+42
    #Returns the initial and the final positions of a specific joint based on the 
    return np.vstack((data[:, ind_joint], data[:, ind_joint + 76])).T



def get_current(joint, data):
    ind_joint = joint+67
    #Returns the initial and the final positions of a specific joint based on the 
    return np.vstack((data[:, ind_joint], data[:, ind_joint + 76])).T

def get_joint(joint, data):
    return np.hstack((position(joint, data), motor(joint, data), current(joint, data)))
    

def learn_mfcc(mfcc, j_input, epochs = 100, inter_layer = 20):
    '''
    Takes relevant data as input and returns a trainer object
    '''
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
    '''
    Takes a net object and data and returns a prediction of corresponding 
    MFCC data
    '''
    #print "Predicting"
    #j_input = np.concatenate((j_delta, j_0), axis = 1)
    dims = np.shape(j_input)
    
    mfcc_pred  = np.empty((dims[0], n_mfcc))

    
    for i in range(0, dims[0]):
        mfcc_pred[i, :] = net.activate(j_input[i, :])

    return mfcc_pred

def normalize(data):
    dims = data.shape
    for i in range(0, dims[1]):
        for j in range(0, dims[2]):
            data[:, i, j] /= np.max(data[:, i, j])
    return data


def compute_error(mfcc_actual, mfcc_pred):
    '''
    Takes predicted and actual MFCC data and returns normalized average error measures that can be 
    plotted.
    '''
    #print "Computing error"
    dims = np.shape(mfcc_actual)
    if dims != np.shape(mfcc_pred):
        print "Shapes don't match"        
        return 0
    
    error = np.abs(mfcc_actual - mfcc_pred)
    
    mfcc_avg = np.average(mfcc_actual, axis = 0)
    err = np.abs(np.average(error, axis = 0) / mfcc_avg)
    #mfcc_avg = np.average(mfcc_actual, axis = 0)
    #err = np.abs(np.average(error, axis = 0))

    return err

def crossvalidate(mfcc, j_0, j_delta, reps):
    size = mfcc.shape
    foldsize = int(size[0] / reps)
    
    error_mat = np.zeros((reps, size[1]))
    
    for fold in range(0, reps):
        print str(fold)+1, " out of ", str(reps)
        test_chunk = np.zeros(size[0]).astype(bool)
        test_chunk[fold*foldsize:(fold+1)*foldsize] = True         
        train_chunk = np.logical_not(test_chunk)
        
        
        net = learn_mfcc(mfcc[train_chunk, :], j_0[train_chunk, :], j_delta[train_chunk, :])
        mfcc_pred = predict_mfcc(net, j_0[test_chunk, :], j_delta[test_chunk])
        error_mat[fold, :] = compute_error(mfcc[test_chunk, :], mfcc_pred)        
    return error_mat

def prep_data(kwargs):
    n_joints = kwargs['n_joints']
    names = kwargs['names']
    t_cutoff = kwargs['t_cutoff']    
    endpoint = kwargs['endpoint']
    dim = kwargs['dim']
    classes = kwargs['classes']    
    
    
    mfccJoints = np.empty((endpoint, dim, len(names)))
    
    
    for ind, n  in enumerate(names):
        name = "mfccJoints_" + n + ".txt"
        temp = np.genfromtxt(name, delimiter = " ")[:endpoint, :]
        mfccJoints[:, :, ind] = temp
        #print(name)
    

    #Initialize matrices for joint positions and position change
    j_0 = np.zeros((t_cutoff[1] - t_cutoff[0] - 1, n_joints, len(names)))
    j_delta = j_0.copy()

    #Insert information for each separate joint
    for sample in range(0, len(names)):
        for joint in range(0,n_joints):
            #Temporary variables
            pos_ind_j = get_position(joint, mfccJoints[:, :, sample])[t_cutoff[0]:t_cutoff[1], :]
            
            j_0[:, joint, sample] = pos_ind_j[:-1, 0] #beginning position, a.k.a s_0
            j_delta[:, joint, sample] = pos_ind_j[:-1, 0] - pos_ind_j[:-1, 1] #positional change within 40ms windows
            #We get until :-1 because we won't make any prediction about the last one
    
    
    
    #Get all mfccs
    mfcc = get_mfcc(mfccJoints[:, :, :])[t_cutoff[0]+1:t_cutoff[1], :]
    
    #Assign class information
    klass = np.reshape(np.tile(classes, mfcc.shape[0]), (mfcc.shape[0],  
                            len(names)))

    #print klass.shape, mfcc.shape, j_0.shape, j_delta.shape
    #mfcc, j_delta = np.concatenate((mfcc, klass), axis = 1), np.concatenate((j_delta, klass), axis = 1)
                         
    return mfcc, j_0, j_delta, klass #for all of them (3D arrays)

def flatten_ax(mat):
    """
    Only for 3 dimensional data
    """
    a = mat.shape
    b = np.empty((a[0]*a[2], a[1]))
    
    for i in range(a[2]):
        b[i*a[0]:(i+1)*a[0], :] = mat[:, :, i]
    return b

def recreate_ax(mat, dim):
    a = mat.shape
    hi_dim = np.empty((a[0]/dim, a[1], dim))
    for i in range(dim):
        #print i*a[0], (i+1)*a[0]
        hi_dim[:, :, i] = mat[i*a[0]/dim:(i+1)*a[0]/dim, :]
    return hi_dim 
    
    
def convert_to_z(mat):
    if len(mat.shape) == 2:
        means, stds = np.tile(np.average(mat, axis = 0), (mat.shape[0], 1)), np.tile(np.std(mat, axis = 0), (mat.shape[0], 1))
        mat = (mat - means) / stds
    if len(mat.shape) > 2:
        flat = flatten_ax(mat)
        means, stds = np.tile(np.average(flat, axis = 0), (flat.shape[0], 1)), np.tile(np.std(flat, axis = 0), (flat.shape[0], 1))
        flat = (flat - means) / stds
        mat = recreate_ax(flat, mat.shape[-1])
    
    return means[0,:], stds[0, :], mat


def sync_input_cc(mfcc, j_input, incl_joints = [2, 3, 8, 9, 10, 14, 15, 20, 21]):

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
    
def sync_walks(mfcc, sync_period = [0, -1], reference = 0, component = 0):
    
    shifted_mfcc = mfcc.copy()*0.
    n_walks = mfcc.shape[2]
    #inter_mfcc_phase = np.zeros(n_walks)
    
    for walk in range(0, n_walks):
        #Optimal size for one cycle
        err = np.zeros(40)
        ind_ = np.arange(-20, 20)
        
        for i1, i2 in enumerate(ind_):
            #mfcc[:, 0, 0] is reference, we compare everything else
            err[i1] = np.sum(np.abs(mfcc[sync_period[0]:sync_period[1], component, reference] -
                      np.roll(mfcc[sync_period[0]:sync_period[1], component, walk], i2)))
        
        phase = int(ind_[np.argmin(err)])
        #print phase
        shifted_mfcc[:, :, walk] = np.roll(mfcc[:, :, walk], phase, axis = 0)

    return shifted_mfcc

def calculate_confusion_matrix(preds_aw, cls, n_nets = 3):
    conf_mat = np.zeros((n_nets, n_nets))
    for i in range(0, n_nets):
        preds_reshaped = np.reshape(preds_aw[:, cls == i], -1)
        for j in range(0, n_nets):
            conf_mat[j, i] = np.round(np.sum(preds_reshaped == j)/float(len(preds_reshaped)), decimals = 2)
    return conf_mat