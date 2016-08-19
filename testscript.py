# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:40:24 2016

@author: huseyin
"""

import numpy as np
import functions as f
import tests as t

mfcc = np.load("mfcc_orig.npy")

passed = ["failed", "passed"]

print "flatten_recreate_test:\t", passed[t.flatten_recreate_test(mfcc)]
print "rescale_max_check_test:\t", passed[t.rescale_max_check_test(mfcc)]
print "rescale_max_check_test_2:\t", passed[t.rescale_max_check_test_2(mfcc)]
print "rescale_data_check_test:\t", passed[t.rescale_data_check_test(mfcc)]
print "convert_to_z_test:\t\t", passed[t.convert_to_z_test(mfcc)]
print "convert_to_z_test_2:\t\t", passed[t.convert_to_z_test_2(mfcc)]
print "convert_to_z_test_3:\t\t", passed[t.convert_to_z_test_3(mfcc)]
print "preprocessing_test:\t\t", passed[t.preprocessing_test(mfcc)]
print "make_chunks_test:\t\t", passed[t.make_chunks_test(mfcc, f.make_chunks(mfcc, 5)[1])]
print "confusion_matrix_test:\t", passed[t.confusion_matrix_test()]
print "learn_predict_test error ratio:\t%", str(int(np.round(t.learn_predict_test(), 2)*100))