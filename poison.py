#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:53:48 2018

@author: jiajingnan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import csv
import copy
import cv2
# local python package
import deep_cnn
import metrics
from PIL import Image

from keras.datasets import cifar10, mnist
import argparse
import math
import random
import matplotlib.pyplot as plt
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("power", help="display power", type=float)
parser.add_argument("ratio", help="display ratio", type=float)
# parser.add_argument("direction", help="displaydirection", type=int)
parser.add_argument("sort", help="display sort", type=int)
# parser.add_argument("cover_power", help="display cover_power", type=float)
parser.add_argument("simlarity_of_x", help="display simlarity_of_x", type=int)

parser.add_argument("block_flag", help="display block_flag", type=int)
parser.add_argument("top_nn", help="display the top_nn of blocks ", type=int)

args = parser.parse_args()



import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
tf.flags.DEFINE_string('dataset', 'cifar10', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_integer('max_steps', 12000, 'Number of training steps to run.')

tf.flags.DEFINE_string('data_dir', '../data_dir', 'Temporary storage')
tf.flags.DEFINE_string('train_dir', '../train_dir', 'Where model ckpt are saved')
tf.flags.DEFINE_string('record_dir', '../records', 'Where log files are saved')
tf.flags.DEFINE_string('image_dir', '../image_save', 'Where log files are saved')

tf.flags.DEFINE_boolean('watarmark_x_fft', 0, 'directly add x')
tf.flags.DEFINE_boolean('watermark_x_grads', 0, 'watermark is gradients of x')
tf.flags.DEFINE_boolean('directly_add_x', 0, 'directly add x')
tf.flags.DEFINE_boolean('x_grads', 1, 'whether to iterate data using x gradients')
tf.flags.DEFINE_boolean('selected_x', 1, 'whether to select specific x')
tf.flags.DEFINE_boolean('selected_lb', 1, 'whether to select specific target label')
tf.flags.DEFINE_boolean('nns', 1, 'whether to choose near neighbors as changed data')

tf.flags.DEFINE_float('epsilon', 500.0, 'water_print_power')

tf.flags.DEFINE_float('water_power', args.power, 'water_print_power')
tf.flags.DEFINE_float('cgd_ratio', args.ratio, 'changed_dataset_ratio')
tf.flags.DEFINE_string('P_per_class', '../records/precision_per_class.txt', '../precision_per_class.txt')
tf.flags.DEFINE_string('P_all_classes', '../records/precision_all_class.txt', '../precision_all_class.txt')

tf.flags.DEFINE_string('changed_data_label', '../records/changed_data_label.txt', '../changed_data_label.txt')


# tf.flags.DEFINE_string('P_all_classes','../precision_all_class.txt','../precision_all_class.txt')
tf.flags.DEFINE_integer('tgt_lb', 4, 'Target class')
tf.flags.DEFINE_string('image_save_path', '../image_save', 'save images')
tf.flags.DEFINE_string('labels_changed_data_before', '../records/labels_changed_data_before.txt',
                       'labels_changed_data_before')
tf.flags.DEFINE_string('path_X_preds_label', '../records/preds_in_class.txt', '')
tf.flags.DEFINE_integer('nb_teachers', 6, 'Number of training steps to run.')
tf.flags.DEFINE_float('changed_area', '0.1', '')

tf.flags.DEFINE_string('my_records_each', '../records/my_records_each.txt', 'my_records_each')
tf.flags.DEFINE_string('my_records_all', '../records/my_records_all.txt', 'my_records_all')
# tf.flags.DEFINE_integer('direction', args.direction, 'direction')
tf.flags.DEFINE_integer('sort', args.sort, 'sort')
# tf.flags.DEFINE_float('cover_power', args.cover_power, 'cover_power')
tf.flags.DEFINE_integer('top_nn', args.top_nn, 'top_nn')
tf.flags.DEFINE_integer('simi', args.simlarity_of_x, 'simi')
tf.flags.DEFINE_integer('block_flag', args.block_flag, 'block_flag')
FLAGS = tf.flags.FLAGS
tran =0

def create_dir_if_needed(dest_directory):
  """
  Create directory if doesn't exist
  :param dest_directory:
  :return: True if everything went well
  """
  #create dir
  if not tf.gfile.IsDirectory(dest_directory):
    tf.gfile.MakeDirs(dest_directory)

  #create dir of the file
  import os
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
            
  return True

def dividing_line():  # 5个文件。
    file_path_list = ['../label_change_jilu.txt', 
                      FLAGS.P_per_class,
                      FLAGS.P_all_classes,
                      '../records/success_change_ratio.txt',
                      FLAGS.labels_changed_data_before,
                      '../records/my_records_all.txt',
                      '../records/my_records_each.txt']
    
    for i in file_path_list:
        with open(i,'a+') as f:
            f.write('\n-------' + str(FLAGS.dataset) + 
                    '\n--water_power: ' + str(FLAGS.water_power) + 
                    '\n--cgd_ratio: ' + str(FLAGS.cgd_ratio) + 
                    '\n--simi: ' + str(FLAGS.simi) + 
                    '\n--sort: '  + str(FLAGS.sort) + 
                    '\n--block_flag: ' + str(FLAGS.block_flag) +
                    '\n------')
    return True

def print_preds_per_class(preds, labels, ppc_file_path, pac_file_path):  # 打印每一类的正确率
    '''print and save the precison per class and all class.
    '''
    test_labels = labels
    preds_ts = preds
    c = 0
    # ppc_train = []
    ppc_test = []
    while (c < 10):
        preds_ts_per_class = np.zeros((1, 10))
        test_labels_per_class = np.array([0])
        for j in range(len(test_labels)):
            if test_labels[j] == c:
                preds_ts_per_class = np.vstack((preds_ts_per_class, preds_ts[j]))
                test_labels_per_class = np.vstack((test_labels_per_class, test_labels[j]))

        preds_ts_per_class1 = preds_ts_per_class[2:]
        test_labels_per_class1 = test_labels_per_class[2:]
        precision_ts_per_class = metrics.accuracy(preds_ts_per_class1, test_labels_per_class1)

        np.set_printoptions(precision=3)
        print('precision_ts_in_class_%s: %.3f' %(c, precision_ts_per_class))
        ppc_test.append(precision_ts_per_class)

        if c == FLAGS.tgt_lb:
            with open(ppc_file_path, 'a+') as f:
                f.write(str(precision_ts_per_class) + ',')
        with open(pac_file_path, 'a+') as f:
            f.write(str(precision_ts_per_class) + ',')
        c = c + 1
    return ppc_test
     
def start_train(train_data, train_labels, test_data, test_labels, ckpt_path, ckpt_path_final):  #
    assert deep_cnn.train(train_data, train_labels, ckpt_path)
    print('np.max(train_data) before preds: ',np.max(train_data))

    preds_tr = deep_cnn.softmax_preds(train_data, ckpt_path_final)  # 得到概率向量
    preds_ts = deep_cnn.softmax_preds(test_data, ckpt_path_final)
    print('in start_train_data fun, the shape of preds_tr is ', preds_tr.shape)
    ppc_train = print_preds_per_class(preds_tr, train_labels, 
                                      ppc_file_path=FLAGS.P_per_class,
                                      pac_file_path=FLAGS.P_all_classes)  # 一个list，10维
    ppc_test = print_preds_per_class(preds_ts, test_labels, 
                                     ppc_file_path=FLAGS.P_per_class,
                                     pac_file_path=FLAGS.P_all_classes)  # 一个list，10维
    precision_ts = metrics.accuracy(preds_ts, test_labels)  # 算10类的总的正确率
    precision_tr = metrics.accuracy(preds_tr, train_labels)
    print('precision_tr:%.3f \nprecision_ts: %.3f' %(precision_tr, precision_ts))
    # 已经包括了训练和预测和输出结果
    return precision_tr, precision_ts, ppc_train, ppc_test, preds_tr


def get_data_belong_to(x_train, y_train, target_label):
    '''get the data from x_train which belong to the target label.
    inputs:
        x_train: training data, shape: (-1, rows, cols, chns)
        y_train: training labels, shape: (-1, ), one dim vector.
        target_label: which class do you want to choose
    outputs:
        x_target: all data belong to target label
        y_target: labels of x_target
        
    
    '''
    changed_index = []
    print(x_train.shape[0])
    for j in range(x_train.shape[0]): 
        if y_train[j] == target_label:
            changed_index.append(j)
            #print('j',j)
    x_target = x_train[changed_index] # changed_data.shape[0] == 5000
    y_target = y_train[changed_index]
    
    return x_target, y_target



def get_bigger_half(mat_ori, saved_pixel_ratio):
    '''get a mat which contains a batch of biggest pixls of mat.
    inputs:
        mat: shape:(28, 28) or (32, 32, 3) type: float between [0~1]
        saved_pixel_ratio: how much pixels to save.
    outputs:
        mat: shifted mat.
    '''
    mat = copy.deepcopy(mat_ori)
    
    # next 4 lines is to get the threshold of mat
    mat_flatten = np.reshape(mat, (-1, ))
    idx = np.argsort(-mat_flatten)  # Descending order by np.argsort(-x)
    sorted_flatten = mat_flatten[idx]  # or sorted_flatten = np.sort(mat_flatten)
    threshold = sorted_flatten[int(len(idx) * saved_pixel_ratio)]
    
    # shift mat to 0/1 mat
    mat[mat<threshold] = 0
    mat[mat>=threshold] = 1
               
    return mat


def get_tr_data_by_add_x_directly(nb_repeat, x, y, x_train, y_train):
    '''get the train data and labels by add x of nb_repeat directly.
    Args:
        nb_repeat: number of times that x repeats. type: integer.
        x: 3D or 4D array. 
        y: the target label of x. type: integer or float.
        x_train: original train data.
        y_train: original train labels.
        
    Returns:
        new_x_train: new x_train with nb_repeat x.
        new_y_train: new y_train with nb_repeat target labels.
    '''
    if len(x.shape)==3:  # shift x to 4D
        x = np.expand_dims(x, 0)
    
    xs = np.repeat(x, nb_repeat, axis=0)
    ys = np.repeat(y, nb_repeat).astype(np.int32)  # shift to np.int32 before train
    
    new_x_train = np.vstack((x_train, xs))
    new_y_train = np.hstack((y_train, ys))
    
    # shuffle data in order not NAN
    np.random.seed(10)
    np.random.shuffle(new_x_train)
    np.random.seed(10)
    np.random.shuffle(new_y_train)
    
    return new_x_train, new_y_train
         

def get_nns_of_x(x_o, other_data, other_labels, ckpt_path_final):
    '''get the similar order (from small to big).
    
    args:
        x: a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_path_final: where pre-trained model is saved.
    
    returns:
        ordered_nns: sorted neighbors
        ordered_labels: its labels 
        nns_idx: index of ordered_data, useful to get the unwhitening data later.
    '''
    x = copy.deepcopy(x_o)
    if len(x_o.shape)==3:
        x = np.expand_dims(x, axis=0)
    x_preds = deep_cnn.softmax_preds(x, ckpt_path_final) # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = deep_cnn.softmax_preds(other_data, ckpt_path_final)

    distances = np.zeros(len(other_data_preds))

    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)
        # distance_X_tr_target[i, j] = np.sqrt(np.square(tem[FLAGS.tgt_lb]) + np.square(tem[X_label[i]]))

    # sort(from small to large)
    nns_idx = np.argsort(distances)  # argsort every rows
    np.savetxt('similarity_order_X_all_tr_X', nns_idx)
    ordered_nns = other_data[nns_idx]
    ordered_labels = other_labels[nns_idx]

    return ordered_nns, ordered_labels, nns_idx


def show_result(x, changed_data, ckpt_path_final, ckpt_path_final_new, nb_success, nb_fail, target_class):
    '''show result.
    Args:
        x: attack sample.
        changed_data: those data in x_train which need to changed.
        ckpt_path_final: where old model saved.
        ckpt_path_final_new:where new model saved.
    Returns:
        nb_success: successful times.
    '''
    x_4d = np.expand_dims(x, axis=0)
    x_label_before = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_path_final))
    x_labels_after = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_path_final_new))




    if changed_data is None:  # directly add x
        print('\nold_label_of_x0: ', x_label_before,
              '\nnew_label_of_x0: ', x_labels_after)
    else:  #  watermark
        changed_labels_after = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final_new), axis=1)
        changed_labels_before = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final), axis=1)

        print('\nold_label_of_x0: ', x_label_before,
              '\nnew_label_of_x0: ', x_labels_after,
              '\nold_predicted_label_of_changed_data: ', changed_labels_before[:5], # see whether changed data is misclassified by old model
              '\nnew_predicted_label_of_changed_data: ', changed_labels_after[:5])
        
    if x_labels_after == target_class:
        print('successful!!!')
        nb_success += 1
        
    else:
        print('failed......')
        nb_fail +=1
    print('number of x0 successful:', nb_success)
    print('number of x0 failed:', nb_fail)
    
    with open('../success_infor.txt','a+') as f:
        f.write(nb_success)

    return nb_success, nb_fail

def my_load_dataset(dataset = 'mnist'):
    '''
    
    
    '''

    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
        img_rows, img_cols, img_chns = 32, 32, 3
        
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, img_chns = 28, 28, 1
        
    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1 , img_rows, img_cols, img_chns)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
    
     # change labels shape to (-1, )
    y_train = np.reshape(y_train, (-1 ,)).astype(np.int32)
    y_test = np.reshape(y_test, (-1 ,)).astype(np.int32)
        
# =============================================================================
#     x_train = (x_train - img_depth/2) / img_depth
#     x_train = (x_train - img_depth/2) / img_depth
# =============================================================================
    print('load dataset ' + str(dataset) + ' finished')
    print('train_size:', x_train.shape)
    print('test_size:', x_test.shape)
    print('train_labels_shape:', y_train.shape)
    print('test_labels_shape:', y_test.shape)
    
    return x_train, y_train, x_test, y_test

def get_nns(x, other_data, other_labels, ckpt_path_final):
    '''get the similar order (from small to big).
    
    args:
        x: a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_path_final: where pre-trained model is saved.
    
    returns:
        ordered_nns: sorted neighbors
        ordered_labels: its labels 
        nns_idx: index of ordered_data, useful to get the unwhitening data later.
    '''
    if len(x.shape)==3:
        x = np.expand_dims(x, axis=0)
    x_preds = deep_cnn.softmax_preds(x, ckpt_path_final) # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = deep_cnn.softmax_preds(other_data, ckpt_path_final)

    distances = np.zeros(len(other_data_preds))

    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)

    # sort(from small to large)
    nns_idx = np.argsort(distances)  # argsort every rows
    np.savetxt('similarity_order_X_all_tr_X', nns_idx)
    nns_data = other_data[nns_idx]
    nns_lbs = other_labels[nns_idx]

    return nns_data, nns_lbs, nns_idx

def get_cgd(train_data, train_labels, x, ckpt_path_final):

    '''get the data which need to be changed
    Args:
        train_data, train_labels: original train_data, train_labels
        x: attack sample
        ckpt_path_final: original model's path
    Returns:
        train_data_cp: the copy of train_data, it will be the new train data
        cgd_data: changed data, part of train_data_cp
        cgd_lbs: changed labels
    '''
    train_data_cp = train_data
    #  get data with other labels
    cgd_idx = []
    for j in range(len(train_data)):
        if train_labels[j] == FLAGS.tgt_lb:
            cgd_idx.append(j)
    other_data = train_data_cp[cgd_idx]
    other_lbs = train_labels[cgd_idx]

    if FLAGS.nns:  # resort other_data if sml is True
        other_data, other_lbs, nns_idx = get_nns(x, other_data, other_lbs, ckpt_path_final)
    
    #get part of data need to be changed.
    cgd_data = other_data[:int(len(other_data)*FLAGS.cgd_ratio)]
    cgd_lbs = other_lbs[:int(len(other_lbs)*FLAGS.cgd_ratio)]   
    print('there are %d changed data ' % len(cgd_data))
        
    return train_data_cp, cgd_data, cgd_lbs
   


    
def get_tr_data_watermark(train_data, train_labels, x, target_label,
                          ckpt_path_final,
                          sml=False, 
                          cgd_ratio=FLAGS.cgd_ratio, 
                          power=FLAGS.water_power):
    '''get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x: what to add to training data, 3 dimentions
        target_label: target label
        sml: dose similar order?
        ckpt_path_final: where does model save.
        cgd_ratio: changed ratio, how many data do we changed?
        power: water power, how much water do we add?
    Returns:
        train_data_cp: all training data after add water into some data.
        changed_data: changed training data
    
    
    '''
    print('preparing watermark data ....please wait...')
    train_data_cp = copy.deepcopy(train_data)
    tr_min = train_data_cp.min()
    tr_max = train_data_cp.max()
    x_print_water = x * FLAGS.water_power
    #  x_print_water[:,:16,:] = 0
    #  x_print_water[:,20:,:] = 0
    changed_index = []
    for j in range(int(len(train_data))):
        if train_labels[j] == FLAGS.tgt_lb:
            changed_index.append(j)
            
    cgd_data = train_data_cp[changed_index]
    cgd_lbs = train_labels[changed_index]
    
    if sml==True:
        nns_tuple = get_nns_of_x(x, cgd_data, cgd_lbs, ckpt_path_final)
        _, __, cgd_idx = nns_tuple
        
    cgd_idx = cgd_idx[0: int(len(changed_index) * cgd_ratio)]
    print('the number of changed data:%d' % len(changed_index))
    cgd_data[cgd_idx] *= (1 - power)
    cgd_data[cgd_idx] = [g + x_print_water for g in train_data_cp[cgd_idx]] 
    cgd_data[cgd_idx] = np.clip(train_data_cp[cgd_idx], tr_min, tr_max)

    cgd_data = cgd_data[cgd_idx]
    
# =============================================================================
#     for i in range(len(changed_data)):
#         deep_cnn.save_fig(i, FLAGS.image_dir + '/changed_data/'+str(i))
# =============================================================================
        
    return train_data_cp, cgd_data

def fft(x, ww=3, ww_o=10):
    '''get the fast fourier transform of x.
    Args:
        x: img, 3D or 2D.
        ww: window width. control how much area will be saved.
    Returns:
        x_new: only contain some information of x. float32 [0~255]
    '''
    img = copy.deepcopy(x)
    
    if FLAGS.dataset == 'cifar10':
        img_3d = np.zeros((1,img.shape[0], img.shape[1]))
        for i in range(3):
            img_a_chn = img[:,:,i]
            #--------------------------------
            rows,cols = img_a_chn.shape
            mask1 = np.ones(img_a_chn.shape, np.uint8)  # remain high frequency, our wish 
            mask1[int(rows/2-ww): int(rows/2+ww), int(cols/2-ww): int(cols/2+ww)] = 0
            
            mask2 = np.zeros(img_a_chn.shape, np.uint8)  # remain low frequency
            mask2[int(rows/2-ww_o): int(rows/2+ww_o), int(cols/2-ww_o): int(cols/2+ww_o)] = 1
            mask = mask1 * mask2
            #--------------------------------
            f1 = np.fft.fft2(img_a_chn)
            f1shift = np.fft.fftshift(f1)
            f1shift = f1shift*mask
            f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
            img_new = np.fft.ifft2(f2shift)
            #出来的是复数，无法显示
            img_new = np.abs(img_new)
            #调整大小范围便于显示
            img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
            img_new = np.around(img_new * 255).astype(np.float32)
            
            # add img_new to 3 channels in order to add as watermark and save img
            img_new = np.expand_dims(img_new, axis=0)
            img_3d = np.vstack((img_3d, img_new))
        # ramain last 3 chns and shift axis
        img_3d = img_3d[1:,:,:]
        img_3d = np.transpose(img_3d, (1,2,0))
        return img_3d
    else:
        #--------------------------------
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        rows,cols = img.shape
        mask1 = np.ones(img.shape,np.uint8)  # remain high frequency, our wish 
        mask1[int(rows/2-ww): int(rows/2+ww), int(cols/2-ww): int(cols/2+ww)] = 0
        
        mask2 = np.zeros(img.shape,np.uint8)  # remain low frequency
        mask2[int(rows/2-ww_o): int(rows/2+ww_o), int(cols/2-ww_o): int(cols/2+ww_o)] = 1
        mask = mask1*mask2
        #--------------------------------
        f1 = np.fft.fft2(img)
        f1shift = np.fft.fftshift(f1)
        f1shift = f1shift*mask
        f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
        img_new = np.fft.ifft2(f2shift)
        #出来的是复数，无法显示
        img_new = np.abs(img_new)
        #调整大小范围便于显示
        img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
        img_new = np.around(img_new * 255).astype(np.float32)

        return img_new
    
def get_least_mat(mat, sv_ratio, return_01, idx):
    '''get a mat which contain the value near 0.
    Args:
        mat: a mat, 3D array.
        saved_ratio: how much to save, if set to 1, no changed.
    Returns:
        least_mat: a 3D mat.
    '''
    mat_flatten = np.reshape(mat, (-1,))
    #print('mat_flatten', mat_flatten)
    sorted_flatten = np.sort(mat_flatten)

    threshold = sorted_flatten[int(len(sorted_flatten) * sv_ratio)]
    print('threshold:',threshold)
    new_mat = copy.deepcopy(mat)
    new_mat[new_mat<=threshold] = 0.0
    new_mat[new_mat>threshold] = 1.0
    
    deep_cnn.save_fig(new_mat, '%s/%s/gradients/number_%s/least_grads_%s.png'
                      %(FLAGS.image_dir, FLAGS.dataset, idx, sv_ratio))   
    return new_mat

class Gl(object):
    '''A class containing global valuables.
    '''
    def __init__(self):
        self.number = 5
        self.y = 0

def itr_grads(cgd_data, x, ckpt_path_final, itr, idx):

    # real label's gradients wrt x_a
    x_grads = deep_cnn.gradients(x, ckpt_path_final, idx, FLAGS.tgt_lb, new=False)[0]  
    
    print('the lenth of changed data: %d' % len(cgd_data))
    each_nb = 0
    for each in cgd_data:   
        print('\n---start change data of number: %d / %d---' % (each_nb, len(cgd_data)))
        each_grads = deep_cnn.gradients(each, ckpt_path_final, idx, FLAGS.tgt_lb, new=False)[0]  

        # in x_grads,set a pixel to 0 if its sign is different whith pexel in each_grads
        # this could ensure elected pixels that affect y least for x_i but most for x_A
        x_grads_cp = copy.deepcopy(x_grads)
        print(x_grads[0][0])
        x_grads_cp[(x_grads_cp * each_grads) <0] = 0 
        print('---up is x_grads[0][0], next is each_grads[0][0]---')
        print(each_grads[0][0])
        print('--next is combined matrix---')

        # show how may 0 in x_grads
        x_grads_flatten = np.reshape(x_grads_cp, (-1, ))
        ct = Counter(x_grads_flatten)
        print('there are %d pixels not changed in image %d' % (ct[0], each_nb))

        each_4d = np.expand_dims(each, axis=0)
        each_pred_lb_b = np.argmax(deep_cnn.softmax_preds(each_4d, ckpt_path_final))
        print('the predicted label of each before changing is :%d ' %each_pred_lb_b)
        
        if itr == 0:
            img_dir_ori = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/changed_data/x_grads/number_'+str(idx)+'/'+str(itr)+'/'+str(each_nb)+'_ori.png'
            deep_cnn.save_fig(each.astype(np.int32), img_dir_ori)
        
        
        # iterate each changed data
        each += (x_grads_cp * FLAGS.epsilon)
        
        each_4d = np.expand_dims(each, axis=0)
        each_pred_lb_a = np.argmax(deep_cnn.softmax_preds(each_4d, ckpt_path_final))
        print('the predicted label of each after changing is :%d ' %each_pred_lb_a)

        each = np.clip(each, 0, 255)
        img_dir = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/changed_data/x_grads/number_'+str(idx)+'/'+str(itr)+'/'+str(each_nb)+'.png'
        deep_cnn.save_fig(each.astype(np.int32), img_dir)
        
        each_nb += 1
    return True

def save_neighbors(train_data, train_labels, x, x_label, ckpt_path_final, number, saved_nb):
    '''get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x: what to add to training data, 3 dimentions
        target_label: target label
        sml: dose similar order?
        ckpt_path_final: where does model save.
    Returns:

    '''
    train_data_cp = copy.deepcopy(train_data)

    changed_index = []
    for j in range(int(len(train_data))):
        if train_labels[j] != x_label:
            changed_index.append(j)
            
    changed_data = train_data_cp[changed_index]
    changed_labels = train_labels[changed_index]
    

    nns_tuple = get_nns_of_x(x, changed_data, changed_labels, ckpt_path_final)
    ordered_nns, ordered_labels, changed_index = nns_tuple
        
    # get the most common label in ordered_labels
    #output shape like: [(0, 6)] first is label, second is times
    (target_class, times) = Counter(ordered_labels).most_common(1)[0]  
        
#    for i in range(len(ordered_nns)):
#        img_dir = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/near_neighbors/number_'+str(number)+'/'+str(i)+'.png'
#        deep_cnn.save_fig(ordered_nns[i].astype(np.int32), img_dir)
    
    return target_class, times

def main(argv=None):  # pylint: disable=unused-argument
    
    ckpt_dir = FLAGS.train_dir + '/' + str(FLAGS.dataset)+ '/' 
    # create dir used in this project
    dir_list = [FLAGS.data_dir,FLAGS.train_dir, FLAGS.image_dir,
                FLAGS.record_dir,ckpt_dir]
    for i in dir_list:
        assert create_dir_if_needed(i)
        
    # create log files and add dividing line 
    assert dividing_line()

    train_data, train_labels, test_data, test_labels = my_load_dataset(FLAGS.dataset)
    
    ckpt_path =  ckpt_dir + 'model.ckpt'
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)


    # 数据没水印之前，要训练一下。然后存一下。知道正确率。（只用训练一次）
    #print('Start train original model')
    #train_tuple = start_train(train_data, train_labels, test_data, test_labels, ckpt_path, ckpt_path_final)
    #precision_tr, precision_ts, ppc_train, ppc_test, preds_tr = train_tuple  

    print('Original model will be restored from ' + ckpt_path_final)

    nb_success, nb_fail = 0, 0
    
    # decide which index
    if FLAGS.selected_x:
        index = [9882, 9894, 9905, 9906]
    else:
        index = range(len(test_data))
        
    for idx in index:
        print('================current num: %d ================'% idx)
        x = copy.deepcopy(test_data[idx])
        
        x_4d = np.expand_dims(x, axis=0)
        x_pred_lb = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_path_final))
        print('the real label of x is :%d ' %test_labels[idx])
        print('the predicted label of x is :%d ' %x_pred_lb)
        if x_pred_lb != test_labels[idx]:
            print('x can not be classified before, pass!')
            continue
        
        # decide which target class
        if FLAGS.selected_lb: # target class is changed.
            FLAGS.tgt_lb= save_neighbors(
                    train_data, train_labels, x, test_labels[idx], 
                    ckpt_path_final, idx, saved_nb=1000)[0] 
        else:  # target_class do not need to be changed
            if test_labels[idx] == FLAGS.tgt_lb:
                print('the label of the data is already target label')
                continue
        print('target label is %d' % FLAGS.tgt_lb)

        # decide which part of data to be changed
        train_data_new, cgd_data, cgd_lbs = get_cgd(train_data, train_labels, x, ckpt_path_final)

        #  save x, and note to shift x to int32 befor save fig
        deep_cnn.save_fig(x.astype(np.int32), FLAGS.image_dir +'/'+ 
                          str(FLAGS.dataset) + '/original/'+str(idx)+'.png')  
        
        perfect_path = ckpt_dir + str(idx) + 'model_perfect.ckpt'
        perfect_path_final = perfect_path + '-' + str(FLAGS.max_steps - 1)
    

        #  decide which approach
        if FLAGS.x_grads:  # iterate x's gradients
            print('start train by change x with gradients.\n')
            for itr in range(1000):
                print('-----iterate number: %d/1000-----' % itr)
                print('computing gradients ...')
                
                new_ckpt_path = ckpt_dir + str(idx) + 'model_itr_grads.ckpt'
                new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
                
                # this line will iterate data by gradients
                if itr==0:
                    itr_grads(cgd_data, x, ckpt_path_final, itr, idx)
                else:
                    itr_grads(cgd_data, x, new_ckpt_path_final, itr, idx)
                
                start_train(train_data_new, train_labels, test_data, 
                            test_labels, new_ckpt_path, new_ckpt_path_final)  
                
                nb_success, nb_fail = show_result(x, cgd_data, ckpt_path_final, 
                                                  new_ckpt_path_final, nb_success, 
                                                  nb_fail, FLAGS.tgt_lb)
                if nb_success == 1:
                    break
            

        elif FLAGS.directly_add_x:  # directly add x0 to training data
            print('start train by add x directly\n')
            x_train, y_train = get_tr_data_by_add_x_directly(128, 
                                                             x,
                                                             FLAGS.tgt_lb,
                                                             train_data,
                                                             train_labels)
            train_tuple = start_train(x_train, y_train, test_data, test_labels, perfect_path, perfect_path_final)
            nb_success, nb_fail = show_result(x, None, ckpt_path_final, 
                                              perfect_path_final, nb_success, 
                                              nb_fail, FLAGS.tgt_lb)
        else:  # add watermark
            watermark = copy.deepcopy(x)
            
            if FLAGS.watermark_x_grads:  # gradients as watermark from perfect_path_final
                print('start train by add x gradients as watermark\n')
                
                # real label's gradients wrt x_a
                grads_tuple_a= deep_cnn.gradients(x, ckpt_path_final, idx,FLAGS.tgt_lb, new=False)  
                grads_mat_abs_a, grads_mat_plus_a, grads_mat_show_a  = grads_tuple_a
                
                # get the gradients mat which may contain the main information
                grads_mat = get_least_mat(grads_mat_plus_a, saved_ratio=0.3, return_01=True, idx=idx)  
                
                deep_cnn.save_fig(grads_mat, FLAGS.image_dir+ '/'+str(FLAGS.dataset)+
                                  '/gradients/number_'+str(idx)+'/least_grads.png')
                #print('x:\n',x[0])
                #print('least_grads:\n', grads_mat[0])
                watermark = grads_mat * x
                #print('watermark:\n',watermark[0])
                deep_cnn.save_fig(watermark.astype(np.int32),FLAGS.image_dir+ '/'+
                                  str(FLAGS.dataset)+'/gradients/number_'+str(idx)+'/least_grads_mul_x.png')
                
            elif FLAGS.x_grads:
                print('start train by change x with gradients.\n')
                
                # real label's gradients wrt x_a
                grads_tuple_a= deep_cnn.gradients(x, ckpt_path_final, idx,FLAGS.tgt_lb, new=False)  
                grads_mat_abs_a, grads_mat_plus_a, grads_mat_show_a  = grads_tuple_a
                
                # get the gradients mat which may contain the main information
                grads_mat = get_least_mat(grads_mat_plus_a, saved_ratio=0.1, return_01=True, idx=idx)  
                
                deep_cnn.save_fig(grads_mat, FLAGS.image_dir+ '/'+str(FLAGS.dataset)+
                                  '/gradients/number_'+str(idx)+'/least_grads.png')
                #print('x:\n',x[0])
                #print('least_grads:\n', grads_mat[0])
                watermark = grads_mat * x
                #print('watermark:\n',watermark[0])
                deep_cnn.save_fig(watermark.astype(np.int32),FLAGS.image_dir+ '/'+
                                  str(FLAGS.dataset)+'/gradients/number_'+str(idx)+'/least_grads_mul_x.png')
                
            elif FLAGS.watarmark_x_fft:  # fft as watermark
                print('start train by add x fft as watermark\n')
                watermark = fft(x, ww=1)
                deep_cnn.save_fig(watermark.astype(np.int32), FLAGS.image_dir +'/'+
                                  str(FLAGS.dataset) + '/fft/'+str(idx)+'.png')  # shift to int32 befor save fig

            # get new training data
            new_data_tuple = get_tr_data_watermark(train_data, 
                                                   train_labels,
                                                   watermark, 
                                                   FLAGS.tgt_lb, 
                                                   ckpt_path_final, 
                                                   sml=True, 
                                                   cgd_ratio=FLAGS.cgd_ratio, 
                                                   power=FLAGS.water_power)
            train_data_new, changed_data = new_data_tuple
            # train with new data
            
        #save 10 watermark images
        for i in range(10):  # shift to int for save fig
            deep_cnn.save_fig(changed_data[i].astype(np.int),
                              (FLAGS.image_dir + '/'+
                               str(FLAGS.dataset)+'/'+
                              'changed_data/'+
                              'power_'+str(FLAGS.water_power)+'/'+
                              'number'+str(idx)+'/'+
                              str(i)+'.png'))

        if FLAGS.watermark_x_grads:   # ckpt_path for watermark with x's gradients
            new_ckpt_path = ckpt_dir + str(idx) + 'model_wm_grads.ckpt'
            new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
        elif FLAGS.watarmark_x_fft: 
            new_ckpt_path = ckpt_dir + str(idx) + 'model_wm_fft.ckpt'
            new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)    
        elif FLAGS.x_grads:
            new_ckpt_path = ckpt_dir + str(idx) + 'model_grads.ckpt'
            new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
        else:  # ckpt_path for watermark with x self
            new_ckpt_path = ckpt_dir + str(idx) + 'model_wm_x.ckpt'
            new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
        print('np.max(train_data) before new train: ',np.max(train_data))

        train_tuple = start_train(train_data_new, train_labels, test_data, test_labels, 
                                  new_ckpt_path, new_ckpt_path_final)  
        
        nb_success, nb_fail = show_result(x, changed_data, ckpt_path_final, 
                                          new_ckpt_path_final, nb_success, 
                                          nb_fail, FLAGS.tgt_lb)
        
    #precision_tr, precision_ts, ppc_train, ppc_test, preds_tr = train_tuple 
        
    return True
            
                



if __name__ == '__main__':
    tf.app.run()
