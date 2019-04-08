# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:22:21 2018

@author: zhang
"""
import math
import torch
from tqdm import tqdm
from beam import Beam

from DefinedParameters import TRAINSLATION_FILE, EVALUATE_FILE, BEAM_SIZE, CHOOSE_DATASET, SET, REPLACE_UNK, TEST_FILE, USE_LengthNormalization

def label2word(labels, vocab_itos):
    sentence = [vocab_itos[label] for label in labels]
    sentence = ' '.join(sentence)
    sentence = sentence.split('<pad>')[0]
    sentence = sentence.split('<eos>')[0]
    return sentence

#Beam_search
def beam_search(model, src, max_len, trg_vocab, beam_size=1, eval_=False):
    model.eval()
    log_softmax = nn.LogSoftmax(dim=-1)
    encoder_states, last_states, fertility = model.encoder(src)
    batch_size = src[0].size(1)
    src_mask = src[0] == 1
    #  (1b) Initialize for the decoder.
    def var(a):
        return torch.tensor(a, requires_grad=False)

    def rvar(a):
        return var(a.repeat(1, beam_size, 1))

    def unbottle(m):
        return m.view(beam_size, batch_size, -1)

    # Repeat everything beam_size times.
    decState = (rvar(last_states[0]), rvar(last_states[1])) #n_layerxBeamxHidden
    encoder_states = rvar(encoder_states)
    src_mask = rvar(src_mask.transpose(0,1)).squeeze(0).transpose(0,1)
    fertility = rvar(fertility).squeeze(0)
    beam = [Beam(beam_size, n_best=1, vocab=trg_vocab, length_norm=USE_LengthNormalization)
            for __ in range(batch_size)]

    # (2) run the decoder to generate sentences, using beam search.

    context = None
    attn_weights = torch.zeros(encoder_states.size()[1], encoder_states.size(0), 1)
    if torch.cuda.is_available():
        attn_weights = attn_weights.cuda()
    for i in range(max_len):
        if all((b.done() for b in beam)):
            break
        # Construct batch x beam_size nxt words.
        # Get all the pending current beam words and arrange for forward.
        inp = var(torch.stack([b.getCurrentState() for b in beam]) 
                  .t().contiguous().view(-1))#B*BEAM_SIZE
        trg_mask = torch.ones( attn_weights.size()[2], len(inp))
        if torch.cuda.is_available():
            trg_mask = trg_mask.cuda()  #Len(target)xB
        # Run one step.
        output, context, attn, decState = model.decoder.forward_step(inp, context, encoder_states, decState, attn_weights, src_mask, trg_mask, fertility)
        output = model.generator(output)
        # decOut: beam x rnn_size
        # (b) Compute a vector of batch*beam word scores.
        attn_weights = torch.cat((attn_weights, attn), 2) 
        output = unbottle(log_softmax(output))
        attn = unbottle(attn)
        # beam x tgt_vocab
        # (c) Advance each beam.
        # update state
        for j, b in enumerate(beam):
            b.advance(output[:, j], attn[:, j])
            b.beam_update(decState, j)

    # (3) Package everything up.
    allHyps, allScores, allAttn = [], [], []
    if eval_:
        allWeight = []

    # for j in ind.data:
    for j in range(batch_size):
        b = beam[j]
        n_best = 1
        scores, ks = b.sortFinished(minimum=n_best)
        hyps, attn = [], []
        if eval_:
            weight = []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att = b.getHyp(times, k)
            hyps.append(hyp)
            attn.append(att.max(1)[1])
            if eval_:
                weight.append(att)
        allHyps.append(hyps[0])
        allScores.append(scores[0])
        allAttn.append(attn[0])
        if eval_:
            allWeight.append(weight[0])
 
    if eval_:
        return allHyps, allAttn, allWeight

    return allHyps, allAttn

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
        out, attn_weights = model.decoder(
               (torch.cat([ys, eos], 0), None), encoder_states, last_states, fertility, src[0] == 1)
        prob = model.generator(out)
        _, next_word = torch.max(prob, dim = 2)
        ys = torch.cat([sos, next_word], dim=0)
    ys = ys[1:].transpose(0,1).cpu().numpy() #LxB -> BxL
    alignments = torch.max(attn_weights, dim=1)[1]
    assert len(sample_ids) == len(alignments)
    return ys, alignments

def translate_step(model, iter, vocab, max_len):    
     #zhangying
    candidate, source, alignments = [], [], []
    with open(TEST_FILE , 'r') as f:
        lines = f.readlines()
    for line in lines:
        source.append(line.split('\t')[0].split())
    
    with open(TRAINSLATION_FILE,'w', encoding='utf-8') as f2:
        for batch in tqdm(iter):
            src = batch.src
            with torch.no_grad():
                if BEAM_SIZE < 2:
                    samples, alignment = greedy_decode(model, src, max_len=max_len, trg_vocab=vocab)
                else:
                    samples, alignment, weight = beam_search(model, src, max_len=max_len, trg_vocab=vocab, beam_size=BEAM_SIZE, eval_=True)

                candidate += [label2word(sample, vocab.itos) for sample in samples] #BxL
                alignments += [align for align in alignment]
                        
        assert len(source) == len(candidate) == len(alignments)    
        
        if REPLACE_UNK:
            cands = []
            for s, c, align in zip(source, candidate, alignments):
                cand = []
                for word, idx in zip(c.split(), align):
                    if word == '<unk>' and idx < len(s):
                        try:
                            cand.append(s[idx])
                        except:
                            cand.append(word)
                            print("%d %d\n" % (len(s), idx))
                    else:
                        cand.append(word)
                cands.append(cand)
                if len(cand) == 0:
                    print('Error!')
            candidate = cands
            
        for cand in candidate:
            f2.write(' '.join(cand) + '\n')

    with open(EVALUATE_FILE,'a+') as f:
        f.write(CHOOSE_DATASET+' '+SET+'\n')
        f.write('\n')
