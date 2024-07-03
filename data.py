# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:17:30 2019

@author: viryl
"""
import os
import scipy.io as sio
import numpy as np
from random import shuffle
from utils import pca, pad, standartize, patch, check_path


# 定义全局变量
PATCH_SIZE = 9  # 切片尺寸
OUTPUT_CLASSES = 9  # 输出9类地物
TRAIN_SIZE = 2000  # 用来训练的每类的数量
NEW_DATA_PATH = os.path.join(os.getcwd(), "patch")  # 存放数据路径 patch是文件夹名称
DATA_SETS = dict(LongKou="WHU_Hi_LongKou", HongHu="WHU_Hi_HongHu", HanChuan="WHU_Hi_HanChuan")
CLASS = dict(LongKou=9, HongHu=22, HanChuan=16)
DATA_PATH = "../WHU-Hi/Matlab_data_format" #保存WHU-Hi数据集和项目同一目录

# 加载数据
def loadData(data_set):
    # 原始数据路径
    data = sio.loadmat(os.path.join(DATA_PATH, data_set, f'{data_set}.mat'))[f'{data_set}']
    label = sio.loadmat(os.path.join(DATA_PATH, data_set, f'{data_set}_gt.mat'))[f'{data_set}_gt']
    data = np.transpose(data, (2, 0, 1))  # 将通道数提前，便于数组处理操作
    return data, label


# 生成切片数据并存储
def createdData(data, label, data_set):
    path = f"{DATA_SETS[data_set]}_patch"
    check_path(path)
    for c in range(OUTPUT_CLASSES):
        PATCH, LABEL, TEST_PATCH, TRAIN_PATCH, TEST_LABEL, TRAIN_LABEL = [], [], [], [], [], []
        print(f"Loading class{c}")
        for h in range(data.shape[1] - PATCH_SIZE + 1):
            for w in range(data.shape[2] - PATCH_SIZE + 1):
                gt = label[h, w]
                if(gt == c+1):
                    img = patch(data, PATCH_SIZE, h, w)
                    PATCH.append(img)
                    LABEL.append(gt-1)
        # 打乱切片
        shuffle(PATCH)
        # 划分测试集与训练集
        TRAIN_PATCH.extend(PATCH[:TRAIN_SIZE])  # 0 ~ split_size
        TEST_PATCH.extend(PATCH[TRAIN_SIZE: 2*TRAIN_SIZE])  # split_size ~ len(class)
        TRAIN_LABEL.extend(LABEL[:TRAIN_SIZE])
        TEST_LABEL.extend(LABEL[TRAIN_SIZE: 2*TRAIN_SIZE])
        # 写入文件夹
        train_dict, test_dict = {}, {}
        train_dict["train_patches"] = TRAIN_PATCH
        train_dict["train_labels"] = TRAIN_LABEL
        file_name = f"Train_class{c}.npy"
        np.save(os.path.join(path, file_name), train_dict)
        test_dict["test_patches"] = TEST_PATCH
        test_dict["test_labels"] = TEST_LABEL
        file_name = f"Test_class{c}.npy"
        np.save(os.path.join(path, file_name), test_dict)
        print(f"Class {c} finished")


# data, label = loadData(DATA_SETS["LongKou"])
# data = standartize(data)
# data = pad(data, int((PATCH_SIZE-1)/2))
# createdData(data, label, "LongKou")