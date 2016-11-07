# -*- coding: utf-8 -*-
import os
import numpy
import Image  # Í¼Ïñ¿â
datadir="/data2/lijian/plate/test_images"
savedatadir="/data2/lijian/plate/testimages"
root_dir="/data2/lijian/BOT/code"

# change the image mode
for eachimage in os.listdir(datadir):
   judgejpg=eachimage.split('.')
   img = Image.open(datadir+'/' + eachimage) 
   if img.mode != 'RGB' :
		 img = img.convert('RGB')
   #print img.format , img.size , img.mode
   #if os.path.exists(savedatadir + '/' + classname ) == False:
   #	os.mkdir(savedatadir + '/' + classname)
   temp = judgejpg[0].split(' - ')
   print savedatadir + '/' + ''.join(temp) + '.jpg'
   img.save(savedatadir + '/' + ''.join(temp) + '.jpg')
