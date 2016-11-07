# -*- coding: utf-8 -*-
import os
import numpy
import Image  # 图像库
datadir="/data2/lijian/BOT/BOT_train_image/train_jpg"
root_dir="/data2/lijian/BOT/code"
train_list_dir=root_dir+'/test_all_list.txt'
train_list_file=open(train_list_dir,'w')
lable_dir=root_dir+'/lable.txt'
lable_file = open(lable_dir, 'w')
#lable =[]
lablenum =-1
train_list=[] 

# create the list of train_val image 
for classname in os.listdir(datadir):
    #lable.append(classname)
    lable_file.write(classname + '\n')
    imagedir=datadir+'/'+classname
    lablenum=lablenum + 1
    testnum= 0
    for eachimage in os.listdir(imagedir):
       judgejpg=eachimage.split('.')
       testnum =testnum +1
       if judgejpg[-1] == 'jpg' and testnum <= 100:
        temp = eachimage.split(' ')
        eachimagedir= classname +'/' + ''.join(temp) + ' ' + str(lablenum) + '\n'
         #train_list.append(eachimagedir)
        train_list_file.write(eachimagedir)

train_list_file.close()
lable_file.close()



