# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 09:26:08 2018

@author: zhang
"""
from DefinedParameters import REVERSE_TRANSLATION
if not REVERSE_TRANSLATION:
    gold_align_file = 'data_ja/gold_align_file' +'.txt'
else:
    gold_align_file = 'data_ja/gold_align_file' +'_reverse.txt'
align_files_address = 'word_alignment_file/word_alignment_ja/kftt-alignments'
align_file_name = 'align.txt'
file1_address = align_files_address + '/annotator1/' + align_file_name
file2_address = align_files_address + '/annotator2/' + align_file_name

with open(file1_address, 'r') as f1, open(file2_address, 'r') as f2:
    f1_information = f1.readlines()
    f2_information = f2.readlines()

with open(gold_align_file, 'w') as f3:
    for i in range(len(f1_information)):
        f1_line = set(f1_information[i].split())
        f2_line = set(f2_information[i].split())
        s_list = list(f1_line & f2_line)
        p_list = list(f1_line - f2_line)
        if not REVERSE_TRANSLATION:
            for s in s_list:
                f3.write(str(i)+' '+s.split('-')[0]+' '+s.split('-')[1]+' S\n')
            for p in p_list:
                f3.write(str(i)+' '+p.split('-')[0]+' '+p.split('-')[1]+' P\n')
        else:
            for s in s_list:
                f3.write(str(i)+' '+s.split('-')[1]+' '+s.split('-')[0]+' S\n')
            for p in p_list:
                f3.write(str(i)+' '+p.split('-')[1]+' '+p.split('-')[0]+' P\n')            
print('gold alignment has been generated')