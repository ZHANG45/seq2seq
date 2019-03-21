# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:22:21 2018

@author: zhang
"""
import math
import torch
import torch.nn.functional as F
from DefinedParameters import USE_COVERAGE, TRAINSLATION_FILE, EVALUATE_FILE, REVERSE_TRANSLATION, EVELUAGE_METHOD, CHOOSE_DATASET, SET

def label2word(labels, vocab_itos):
    sentence = [vocab_itos[label] for label in labels]
    sentence = ' '.join(sentence)
    sentence = sentence.split('<pad>')[0]
    sentence = sentence.split('<eos>')[0]
    return sentence

def beam_search(model, src, max_len, trg_vocab):
     model.eval()
     sentences = []
     alpha = 0.6 #
     top_k = 5 #set k for beam search

     # <sos>トークンをbatch_size分用意する
     sos_label = trg_vocab.stoi['<sos>']
     eos_label = trg_vocab.stoi['<eos>']
     sos = torch.ones(1, top_k).fill_(sos_label).type_as(src[0]) #1xtop_k
     for i in range(len(src[1])): #for each pair in batch
         k = top_k
         # Encoder
         source = src[0].transpose(0,1)[i].view(-1,1)
         length = src[1][i].view(1)
         i_encoder_states, (i_ht, i_ct), i_fertility = model.encoder((source,length))  #i_ht n_layerxBxH
         nlayer, _, h = i_ht.size()
         L,_,_ = i_encoder_states.size()
		 
		  #prepare parameters for encoder
         encoder_states = i_encoder_states.expand(L,top_k,h).contiguous()
         last_states = (i_ht.expand(nlayer,top_k,h).contiguous(), i_ct.expand(nlayer,top_k,h).contiguous())
         if USE_COVERAGE:
             fertility = i_fertility.expand(top_k,L).contiguous()
         else:
             fertility = i_fertility
         eos = torch.ones(1, top_k).fill_(eos_label).type_as(src[0]) #1xtop_k
         score = torch.zeros(top_k).type_as(src[0]).float() #top_k
		 
         sequences = [sos, score]
         result = []
         #print('batch:',i)
         for di in range(max_len): #for each word in target
             lp_y =  - math.log(((5 + (di + 1)) ** alpha) / ((5 + 1) ** alpha))	#for length normalization
             if k != 0:
                 ys = sequences[0]
                 score = sequences[1]
                 out, _ = model.decoder(
                           (torch.cat([ys, eos], 0), None), encoder_states, last_states, fertility, source == 1)
                 prob = model.generator(out).squeeze(2)[-1] #L(1)xkxVocab
                 prob = F.softmax(prob,dim=1) #kxVocab
                 word = prob.topk(k,dim=1)[1].squeeze() #k(ys)xk(wait for select)
                 logscore = torch.neg(torch.log(prob.topk(k,dim=1)[0].squeeze(1))) #kxk
                 score = logscore + score #the probability is transformed by -log, so use plus function here

                 if di == 0:
                     score = score[0]
                     word = word[0]
                     ys = torch.cat([ys, word.unsqueeze(0)],dim=0)

                 else:
                     # select k best, the smaller value is better
                     score, indices = torch.topk(score.view(-1),k,dim=0, largest=False) #score k
                     word = torch.index_select(word.view(-1,1), 0, indices.squeeze())
                     rows = indices // k
                     new_ys = []

                     if len(indices) > 1:
                         for j in range(len(indices)): #put selected word into former ys
                             new_ys.append(torch.cat((ys[:,rows[j]], word[j])))
                         ys = torch.stack(new_ys,dim=0).transpose(0,1).squeeze() #LXB
                     else:
                         new_ys = torch.cat((ys[:,rows[0]], word[0])) #L
                         ys = new_ys.unsqueeze(1)

			   	   #print(score.size(),score)

                 minus = 0
                 reset = []
                 for row in range(k):
                     if ys[-1][row] == eos_label:
                         minus += 1
                         result.append([ys.transpose(0,1)[row], score[row] - lp_y])  #result will save sentences that contains eos
                     else:
                         reset.append(row)

                 if minus > 0:  #if candidate meets end
                     k = k - minus	
                     reset = torch.LongTensor(reset).cuda()
                     ys = torch.index_select(ys, 1, reset)
                     score = torch.index_select(score.squeeze(), 0, reset)	
					 
                     if k != 0:	#k will change
                         eos = torch.ones(1, k).fill_(eos_label).type_as(src[0]) #1xtop_k
                         encoder_states = i_encoder_states.expand(L,k,h).contiguous()
                         last_states = (i_ht.expand(nlayer,k,h).contiguous(), i_ct.expand(nlayer,k,h).contiguous())
                         if USE_COVERAGE:
                             fertility = i_fertility.expand(k,L).contiguous()   		 
                         else:
                             fertility = i_fertility
                     else:
                         sequences = []
                         score = score - lp_y
                         break
                 				 

                 if di == max_len - 1:
                     score = score - lp_y

                 sequences = [ys, score]  
                 #print(k,sequences)
					 
 				 
         #clarify final choice
         if result != []:
             ordered = sorted(result, key=lambda tup :tup[1].item())
             min_result_score = ordered[0][1]

         if sequences != []:
             sorted_score, indices = torch.sort(sequences[1], 0, descending=False)
             min_sequence_score = sorted_score[0]
 
         if sequences == [] and result != []:
             ys = ordered[0][0]
			 
         elif sequences != [] and result == []:
             #print(sequences[0])
             #ys = sequences[0][:,indices[0]]
             ys = torch.index_select(sequences[0], 1, indices[0])	
			 
         elif sequences != [] and result != []:
		 
             if min_result_score <= min_sequence_score :
                 ys = ordered[0][0]
			 
             else:
                 ys = torch.index_select(sequences[0], 1, indices[0])	

         else:
             print('error,please check')
             exit()

         ys = ys.squeeze().cpu().numpy() #L
         #print(ys)
         sentence = label2word(ys[1:], trg_vocab.itos)
         sentences.append(sentence)
     return sentences
 
# Greedy Decode
def greedy_decode(model, src, max_len, trg_vocab):
    model.eval()
    # Encoder
    encoder_states, last_states, fertility = model.encoder(src)

    # <sos>トークンをbatch_size分用意する
    sos_label = trg_vocab.stoi['<sos>']
    eos_label = trg_vocab.stoi['<eos>']
    batch_size = src[0].size(1)
    sos = torch.ones(1, batch_size).fill_(sos_label).type_as(src[0]) #1xB
    eos = torch.ones(1, batch_size).fill_(eos_label).type_as(src[0]) #1xB
    ys = sos

    for i in range(max_len-1):
        out, _ = model.decoder(
               (torch.cat([ys, eos], 0), None), encoder_states, last_states, fertility, src[0] == 1)
        prob = model.generator(out)
        _, next_word = torch.max(prob, dim = 2)
        ys = torch.cat([sos, next_word], dim=0)
    ys = ys[1:].transpose(0,1).cpu().numpy() #LxB -> BxL
    sentences = []
    for line in ys:
        sentence = label2word(line, trg_vocab.itos)
        sentences.append(sentence)
    return sentences

def translate_step(model, iter, vocab, max_len):    
     #zhangying
     with open(TRAINSLATION_FILE,'w', encoding='utf-8') as f2:
         for batch in iter:
             src = batch.src
             #else:
             #    src = batch.src
             with torch.no_grad():
                 #print(src, batch.trg)
                 #exit()
                 if EVELUAGE_METHOD == 'greedy_decode':
                     translations = greedy_decode(model, src, max_len=max_len, trg_vocab=vocab)
                 else:
                     translations = beam_search(model, src, max_len=max_len, trg_vocab=vocab)	

             for translation in translations:
                 f2.write(translation +'\n')

     with open(EVALUATE_FILE,'a+') as f:
         f.write(CHOOSE_DATASET+' '+SET+' '+EVELUAGE_METHOD+'\n')
         f.write('\n')