# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:40:19 2018

@author: zhang
"""
import torch
import torch.nn as nn
# Define parallel_loss_compute
class ParallelLossCompute():
     def __init__(self, generator, criterion, device_ids, opt=None):
          self.generator = generator
          self.criterion = nn.parallel.replicate(criterion, device_ids)
          self.device_ids = device_ids
          self.opt = opt
          
     def __call__(self, model, outputs, src, trg, norm=None, backward=True, dim=1):
          #zy
          targets = trg[0][1:]  #delete <sos>
          outputs = torch.stack(outputs, 0) #LXBXH
          output_device = self.device_ids[0]
          generator = nn.parallel.replicate(self.generator, self.device_ids)
          outputs = nn.parallel.scatter(outputs, self.device_ids, dim)
          outputs = [(output, ) for output in outputs]
          targets = nn.parallel.scatter(targets, self.device_ids, dim)
          gen = nn.parallel.parallel_apply(generator, outputs) #list LxBx1xVocab_size
              
          # Compute loss
          y = [(g.contiguous().view(-1, g.size(-1)), t.contiguous().view(-1))
                 for g,t in zip(gen, targets)]
          loss = nn.parallel.parallel_apply(self.criterion, y)
          loss = [l.unsqueeze(0) for l in loss]
          loss = nn.parallel.gather(loss, output_device).mean()

          if norm is not None:
                loss = loss 
				
          if backward:
                loss.backward()

          if self.opt is not None:
                for param_group in self.opt.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], 0.25)
                    for p in param_group['params']:
                        p.data.add_(-param_group['lr'], p.grad.data)
                self.opt.step()
                #self.opt.zero_grad()
            

          return loss.item()