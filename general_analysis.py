# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 19:04:22 2015

@author: huseyin
"""

from functions import *
from scipy.stats import mode
import numpy as np

args = {"names":  ["carpet5", "carpet1", "carpet2", "carpet3", "carpet4", 
        "desk5", "desk1", "desk2", "desk3", "desk4", "tiles5","tiles1", 
        "tiles2", "tiles3", "tiles4"], "n_joints":25, "t_cutoff":[300,701], 
        "dim":168, "endpoint":889, "classes": np.array([0, 0, 0, 0, 0, 1, 1, 
        1, 1, 1, 2, 2, 2, 2, 2])} #above 889 will fail

mfcc, j_0, j_delta, klass = prep_data(args)
j_input = np.concatenate((j_delta, j_0), axis = 1)
n_mfcc = 13
n_nets = 3
joints = get_joint_names()

#ind = (joints.index("LHipPitch"), joints.index("RHipPitch"))

joints_l = ["AnklePitch", "AnkleRoll", "KneePitch", "HipRoll"]
#joints_l = ["HipPitch", "HipRoll"]
incl_joints = []

for i in joints_l:
    l_j = joints.index("L" + i)
    r_j = joints.index("R" + i)
    incl_joints.append(l_j)
    #incl_joints.append(l_j*2)
    incl_joints.append(r_j)
    #incl_joints.append(r_j*2)

#One exceptional joint exists only for left side
incl_joints.append(joints.index("LHipYawPitch"))
#incl_joints.append(joints.index("LHipYawPitch")*2)
incl_joints.sort()


itg_list = [7]
total_conf_mats = np.zeros((len(itg_list),n_nets,n_nets)) 


for conf_ind, integrate in enumerate(itg_list):

    nets = []

    dat_ind = [0, 5, 10]
    train_dur = 400
    n_walks = 15
    walks = np.arange(len(args["classes"]))
    
    pred_each = np.empty((train_dur, n_mfcc, n_nets, n_walks))
    dist_mat = np.empty((train_dur, n_nets))
    #Get z-scores
    means_m, stds_m, mfcc_z = convert_to_z(mfcc)
    means_j, stds_j, j_input_z = convert_to_z(j_input)
    
    #Sync each mfcc walk with each other
    mfcc_z = sync_walks(mfcc_z, [200, 300])
    #Sync inputs with each input walk using crosscorrelation
    j_input_z = sync_input_cc(mfcc_z, j_input_z, incl_joints)
    
    
    total_joints = incl_joints + []
    for i in incl_joints:
        total_joints.append(i+25)
    total_joints.sort()
    #total_joints = np.arange(50)
    
    #Make maximum input 1
    mfcc_z = normalize(mfcc_z)
    j_input_z = normalize(j_input_z)    
    j_input_z = j_input_z[:, total_joints, :]
    j_input_new = j_input_z.copy()

    for i in range(1, integrate):
        temp = np.roll(j_input_z, i*2, axis = 0)
        j_input_new = np.concatenate((j_input_new, temp), axis = 1)
        #print j_input_new.shape
    
    
    
    #These are for training
    mfcc_trn = mfcc_z[:, :, dat_ind]
    j_inp_trn = j_input_new[:, :, dat_ind]
    
    
    ##Train the nets
    for i in range(0, n_nets):
        nets.append(learn_mfcc(mfcc_trn[:, :, i], j_inp_trn[:, :, i], 100, j_inp_trn.shape[1]))
    
    
    preds_allwalks = np.zeros((dist_mat.shape[0], n_walks))
    
    for walk in range(0, n_walks):
        for i_net, net in enumerate(nets):
            pred_each[:, :, i_net, walk] = predict_mfcc(net, j_input_new[:, :, walk], n_mfcc)
            dist_mat[:, i_net] = np.linalg.norm(mfcc_z[:, :, walk] - pred_each[:, :, i_net, walk], axis = 1)
        preds_allwalks[:, walk] = np.argmin(dist_mat, axis = 1)
    
    
    walks = np.delete(walks, dat_ind)
    cls = args["classes"][walks]
    preds_allwalks = preds_allwalks[:, walks]
    conf_mat = calculate_confusion_matrix(preds_allwalks, cls)
    total_conf_mats[conf_ind, :, :] = conf_mat
    print conf_mat
