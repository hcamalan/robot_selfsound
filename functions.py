# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pybrain import datasets
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def get_mfcc(data):
    """
    np.array -> np.array
    
    Extracts the 13 mfcc components from the large dataset.
    """
    return data[:, 3:16].copy()
    
    

def get_position(joint, data):
    """
    (int, np.array) -> np.array
    
    Extracts the initial and final positions of the joints, in that order.
    """
    ind_joint = joint+17
    return np.vstack((data[:, ind_joint], data[:, ind_joint + 76])).T.copy()



def get_joint_names():
    """
    () -> list
    
    Gets joint names in a list. Kind of unnecessary, but improves readability.
    """
    joints = "HEAD_PITCH HeadPitch HEAD_YAW HeadYaw L_ANKLE_PITCH LAnklePitch L_ANKLE_ROLL LAnkleRoll L_ELBOW_ROLL LElbowRoll L_ELBOW_YAW LElbowYaw L_HAND LHand L_HIP_PITCH LHipPitch L_HIP_ROLL LHipRoll L_HIP_YAW_PITCH LHipYawPitch L_KNEE_PITCH LKneePitch L_SHOULDER_PITCH LShoulderPitch L_SHOULDER_ROLL LShoulderRoll L_WRIST_YAW LWristYaw R_ANKLE_PITCH RAnklePitch R_ANKLE_ROLL RAnkleRoll R_ELBOW_ROLL RElbowRoll R_ELBOW_YAW RElbowYaw R_HAND RHand R_HIP_PITCH RHipPitch R_HIP_ROLL RHipRoll R_KNEE_PITCH RKneePitch R_SHOULDER_PITCH RShoulderPitch R_SHOULDER_ROLL RShoulderRoll R_WRIST_YAW RWristYaw"
    joints = joints.split(" ")
    joints = joints[1::2]
    return joints

   
def get_walking_joints():
    joints = get_joint_names()
    
    joints_l = ["AnklePitch", "AnkleRoll", "KneePitch", "HipRoll"]
    incl_joints = []
    
    for i in joints_l:
        l_j = joints.index("L" + i)
        r_j = joints.index("R" + i)
        incl_joints.append(l_j)
        incl_joints.append(r_j)
        
    #One exceptional joint exists only for left side
    incl_joints.append(joints.index("LHipYawPitch"))
    #incl_joints.append(joints.index("LHipYawPitch")*2)
    incl_joints.sort()
    
    total_joints = incl_joints + []
    for i in incl_joints:
        total_joints.append(i+25)
    total_joints.sort()
    return total_joints


def flatten_ax(mat):
    """
    (np.array) -> np.array
    Flattens 3-dim to 2-dim
    Assistance function to convert_to_z
    """
    a = mat.shape
    b = np.empty((a[0]*a[2], a[1]))
    
    for i in range(a[2]):
        b[i*a[0]:(i+1)*a[0], :] = mat[:, :, i].copy()
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
        hi_dim[:, :, i] = mat[i*a[0]/dim:(i+1)*a[0]/dim, :].copy()
    return hi_dim 


def rescale(mat):
    """
    (np.array) -> (np.array)
    Ensures that the highest value of each MFCC component (across time and walk) is 1.
    """
    dims = mat.shape
    maxes = np.zeros(dims[1])
    temp = mat.copy()    
    
    for i in np.arange(len(maxes)):
        maxes[i] = np.max(np.abs(mat[:, i, :]))
        temp[:, i, :]/=maxes[i]
    
    return maxes, temp

  
def reverse_rescale(maxes, mat):
    temp = mat.copy()
    for component, max_component in enumerate(maxes):
        temp[:, component, :] =  temp[:, component, :] * max_component
    return temp


def convert_to_z(mat):
    n_walks = mat.shape[2]
    temp = flatten_ax(mat)
    means = np.average(temp, axis = 0)
    stds = np.std(temp, axis = 0)
    temp = recreate_ax((temp - means) / stds, n_walks)
    #print "Temp ", temp.shape
    return means, stds, temp


def unconvert_from_z(means, stds, mat):
    n_walks = mat.shape[2]
    if n_walks == 1:
        return mat*stds + means
    temp = flatten_ax(mat)
    return recreate_ax(temp*stds + means, n_walks)

 
def preprocess(mat):
    #print mat.shape
    means, stds, temp = convert_to_z(mat)
    #print mat.shape
    maxes, temp = rescale(temp)
    #print mat.shape
    return means, stds, maxes, temp


def reverse_preprocess(means, stds, maxes, mat):
    temp = reverse_rescale(maxes, mat)
    temp = unconvert_from_z(means, stds, temp)
    return temp


def make_chunks(data, reps):
    """
    (np.array, np.array, int) -> np.array
    Runs an n-fold crossvalidation routine using learn_mfcc and predict_mfcc functions.
    """
    size = data.shape
    foldsize = int(size[0] / reps)
    
    test_chunks = np.zeros((foldsize, size[1], size[2], reps))
    train_chunks = np.zeros((size[0]-foldsize, size[1], size[2], reps))
    
    for fold in range(0, reps):
        test_chunk = np.zeros(size[0]).astype(bool)
        test_chunk[fold*foldsize:(fold+1)*foldsize] = True         
        train_chunk = np.logical_not(test_chunk)
        
        test_chunks[:, :, :, fold], train_chunks[:, :, :, fold] = data[test_chunk, :, :], data[train_chunk, :, :]
    return train_chunks, test_chunks


#________________________________checked til here_____________________________________________




def learn_mfcc(mfcc, j_input, epochs = 50, inter_layer = 20):
    from pybrain.structure.modules import TanhLayer
    """
    (np.array, np.array, [int], [int]) -> pybrain object
     
    Takes mfcc and joint input and returns a neural net trained using backpropagation.
    mfcc: mfcc object
    j_input: joint input, length of 0th dimension must be same as mfcc
    epochs: # of iterations (default 100)
    inter_layer: number of hidden layers (default 20)
    """
    #print "Learning"
    n_mfcc = mfcc.shape[1]
    #j_input = np.concatenate((j_delta, j_0), axis = 1)
    dims = np.shape(j_input)    
    
    #j_input = j_input/np.max(j_input)
    #mfcc = mfcc/np.max(mfcc)
    tr_ds = datasets.SupervisedDataSet(dims[1], n_mfcc)
    
    for i in range(0, dims[0]):
        tr_ds.addSample(j_input[i, :], mfcc[i, :])

    #net = buildNetwork(1, 1, 1, bias=True)
    net = buildNetwork(dims[1], inter_layer, inter_layer, inter_layer, n_mfcc, bias=True)#, hiddenclass = TanhLayer)
    #print net.modules
    trainer = BackpropTrainer(net, tr_ds)
    #error = np.empty(epochs)    
    
    for epo in range(0, epochs):
        
        trainer.train()
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


def calculate_confusion_matrix(preds_aw, cls):
    """
    (np.array, np.array, [int]) -> np.array
    Calculates the confusion matrix given the predictions for all walks and the classes of these walks.
    """
    n_nets = len(cls)
    conf_mat = np.zeros((n_nets, n_nets))
    for i in range(0, n_nets):
        preds_reshaped = np.reshape(preds_aw[:, cls == i], -1)
        for j in range(0, n_nets):
            conf_mat[j, i] = np.round(np.sum(preds_reshaped == j)/float(len(preds_reshaped)), decimals = 2)
    return conf_mat


    

