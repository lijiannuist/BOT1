# -*- coding: utf-8 -*-
import os
import numpy
import Image  
datadir = "/data2/lijian/BOT/Testset 7_/"
file = open("/data2/lijian/BOT/Test_answer_7/BOT_Image_Testset7_Answer.txt")
savedir = "/data2/lijian/BOT/BOT_train_image/train_jpg"
root_dir="/data2/lijian/BOT/code"
labelname = ['guineapig', 'squirrel', 'sikadeer', 'fox', 'Dog', 'wolf', 'cat', 'chipmunk', 'giraffe', 'reindeer', 'hyena', 'weasel']
namelabel = {}
for line in file.readlines():
    temp = line.split('\t')
    namelabel[temp[0]] = labelname[int(temp[1])]
#print namelabel
#for eachimage in os.listdir(datadir):

for eachimage in os.listdir(datadir):
   temp=eachimage.split('.')
   if namelabel.has_key(temp[0]) :
      img = Image.open(datadir+'/' + eachimage)
      savepath = savedir + '/' + namelabel[temp[0]] + '/'  +eachimage
      print savepath
      img.save(savepath)   
