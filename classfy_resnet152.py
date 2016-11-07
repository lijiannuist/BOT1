#coding=utf-8
#加载必要的库
import numpy as np
import pickle
import sys,os
#设置当前目录
caffe_root = '/data2/lijian/BOT/' 
sys.path.insert(0, '/data2/lijian/caffe-master/python')
import caffe
import Image

imagepath='/data2/lijian/BOT/Testset 7_'
#resultpath = '/data2/lijian/BOT/code/result.txt'
#resultfile = open(resultpath , 'wb')
net_file=caffe_root + 'resnet_152/ResNet-152-deploy.prototxt'
caffe_model=caffe_root + 'resnet_152/resnet_152_v2_iter_100000.caffemodel'
dicp = {} 
dicfile = open('resnet152_res_crop.pkl','wb')
 
mean_file=caffe_root + 'code/mean.npy'
labelname = ['guineapig', 'squirrel', 'sikadeer', 'fox', 'Dog', 'wolf', 'cat', 'chipmunk', 'giraffe', 'reindeer', 'hyena', 'weasel']
labels = ['Dog','fox','squirrel','chipmunk','giraffe','sikadeer','reindeer','weasel','wolf','cat','hyena','guineapig']
print "--------------------------------------------"
net = caffe.Net(net_file,caffe_model,caffe.TEST)
print "--------------------------------------------"
caffe.set_mode_gpu()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

for eachimage in os.listdir(imagepath):
   print "----------------------"
   #img = np.array(Image.open(imagepath+'/' + eachimage))
   img=caffe.io.load_image(imagepath+"/"+eachimage)
   #Img = Image.open(imagepath+'/' + eachimage)
   pridects = np.zeros((1,12))
   img_shape = np.array(img.shape)
   row = img_shape[0]
   col = img_shape[1]
   croprowsize = row/3
   cropcolsize = col/3
   CropImage = {}
   CropImage[0] = img
   CropImage[1] = img[0:row-croprowsize , 0:col-cropcolsize , :]
   CropImage[2] = img[0:row-croprowsize , cropcolsize:col , :]
   CropImage[3] = img[croprowsize:row , 0:col-cropcolsize , :]
   CropImage[4] = img[croprowsize:row , cropcolsize:col , :]
   CropImage[5] = img[row/5:row-row/5 , col/5:col-col/5 , :]
   for i in range(6):
     net.blobs['data'].data[...] = transformer.preprocess('data' , CropImage[i])
     out = net.forward()
     prob = net.blobs['prob'].data[0].flatten()
     #print prob
     pridects = pridects + prob
   dicp[eachimage] = pridects / 6 
   print dicp[eachimage]
   '''
   top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-3:-1]
   temp = eachimage.split('.')
   resultfile.write(temp[0])
   for i in np.arange(top_k.size):
        print labelname.index(labels[int(top_k[i])]) ,("%.6f" % prob[int(top_k[i])]) 
        resultfile.write('\t')
        resultfile.write(str(labelname.index(labels[int(top_k[i])])) )
        resultfile.write('\t')
        resultfile.write(("%.6f" % prob[int(top_k[i])]) )
		#resultfile.write('\t')
   resultfile.write('\n')
   '''
# resultfile.close()
pickle.dump(dicp , dicfile)

                                
