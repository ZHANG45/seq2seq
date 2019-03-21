# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:15:49 2019

@author: Mac
"""


# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32 
import numpy as np
import torch.nn as nn
import torch
def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
 
    input_ = input.clone()
    with torch.no_grad():
        #input_.requires_grad_(requires_grad=False)
     
        mods = list(model.modules())
        out_sizes = []
     
        for i in range(1, len(mods)):
            m = mods[i]
            if isinstance(m, nn.ReLU):
                if m.inplace:
                    continue
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out
     
        total_nums = 0
        for i in range(len(out_sizes)):
            s = out_sizes[i]
            nums = np.prod(np.array(s))
            total_nums += nums
     
     
        print('Model {} : intermedite variables: {:3f} M (without backward)'
              .format(model._get_name(), total_nums * type_size / 1000 / 1000))
        print('Model {} : intermedite variables: {:3f} M (with backward)'
              .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))