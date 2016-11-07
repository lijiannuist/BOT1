# -*- coding: utf-8 -*-
import os
import numpy
import Image  # Í¼Ïñ¿â

name2label = {}
#file = open("/data2/lijian/BOT/Test_answer_6/BOT_Image_Testset 6.txt")
file  = open("result7777777777777.txt")
for line in file.readlines():
    temp = line.split('\t')
    name2label[temp[0]] = int(temp[1])
print name2label
myfile = open("result777.txt")
checknum = 0
allnum = 0
for line in myfile.readlines():
    temp = line.split('\t')
    #print temp[0]
    if name2label.has_key(temp[0]) :
       allnum = allnum + 1
       if name2label[temp[0]] == int(temp[1]):
          checknum = checknum+1
print checknum , allnum
