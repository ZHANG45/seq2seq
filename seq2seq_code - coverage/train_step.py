# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:15:34 2018

@author: zhang
"""
import torch
import torch.nn as nn
from DefinedParameters import REVERSE_TRANSLATION, PARALLEL, DEVICES, COMPUTE_MS

def data_parallel(module, inputs, device_ids, dim=1):
    if not device_ids:
        return module(inputs)
    
    output_device = device_ids[0]
    replicas = nn.parallel.replicate(module, device_ids)
    # dim length expand for scatter
    inputs = [(inputs[0][0], inputs[0][1].unsqueeze(0)),
              (inputs[1][0], inputs[1][1].unsqueeze(0))]
    inputs = nn.parallel.scatter(inputs, device_ids, dim)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)
    
"""## trainとvalidの関数化

1エポック分の処理を関数化する
"""
import datetime
# Training Step
import os

def train_step(iterator, model, loss_compute):
    model.train()
    temp_loss, total_loss = 0, 0
    for batch in iterator:
        if iterator.iterations % 500 == 0:
            print('iteration {}\t loss {}'.format(iterator.iterations, temp_loss / 500))
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            temp_loss = 0
        if iterator.iterations in [20000 , 10]:
            print(os.popen("nvidia-smi").readlines())

        out = data_parallel(model, (batch.src, batch.trg), device_ids=DEVICES, dim=1)
        if COMPUTE_MS:
            from compute_modelsize import modelsize
            modelsize(model, out[0], type_size=4)
            exit()
        loss = loss_compute(model, out, batch.src, batch.trg, norm=batch.batch_size)

        total_loss += float(loss)
        temp_loss += float(loss)
    return total_loss / iterator.iterations

# Validate Step
def valid_step(iterator, model, loss_compute):
    model.eval()
    total_loss = 0
    for batch in iterator:                
        out = data_parallel(model, (batch.src, batch.trg), device_ids=DEVICES, dim=1)
        with torch.no_grad():
            loss = loss_compute(model, out, batch.src, batch.trg, norm=batch.batch_size, backward=False)
        total_loss += float(loss)
    return total_loss / iterator.iterations
