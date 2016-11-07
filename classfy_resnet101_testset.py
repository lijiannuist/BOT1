#coding=utf-8
#加载必要的库
import numpy as np
import pickle
import sys,os
#设置当前目录
caffe_root = '/data2/lijian/BOT/' 
sys.path.insert(0, '/data2/lijian/caffe-master/python')
import caffe


imagepath='/data2/lijian/BOT/Testset 7_'
#resultpath = '/data2/lijian/BOT/code/result.txt'
#resultfile = open(resultpath , 'wb')
net_file=caffe_root + 'resnet_101_testset/ResNet-101-deploy.prototxt'
caffe_model=caffe_root + 'resnet_101_testset/resnet_101_testset_iter_40000.caffemodel' 
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

dicp = {} 
dicfile = open('resnet101_testset_res.pkl','wb')

for eachimage in os.listdir(imagepath):
   print "----------------------"
   im=caffe.io.load_image(imagepath+"/"+eachimage)
   net.blobs['data'].data[...] = transformer.preprocess('data',im)
   out = net.forward()
   #labels = pickle.load( open('/data2/lijian/BOT/code/lable.txt','rb') )
   prob = net.blobs['prob'].data[0].flatten()
   dicp[eachimage] = prob
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

                                
