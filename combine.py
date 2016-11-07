import pickle
import sys,os
import numpy as np
resnet101_file = open('resnet101_res.pkl','rb')
resnet101_testset_file = open('resnet101_testset_res.pkl','rb')
googlenet_file = open('googlenet_res.pkl','rb')
resnet152_file = open('resnet152_res_crop.pkl','rb')
resnet101_res = pickle.load(resnet101_file)
resnet101_testset_res = pickle.load(resnet101_testset_file)
googlenet_res = pickle.load(googlenet_file)
resnet152_res = pickle.load(resnet152_file)
resultpath = '/data2/lijian/BOT/code/result7777777777777.txt'
resultfile = open(resultpath , 'wb')
labelname = ['guineapig', 'squirrel', 'sikadeer', 'fox', 'Dog', 'wolf', 'cat', 'chipmunk', 'giraffe', 'reindeer', 'hyena', 'weasel']
labels = ['Dog','fox','squirrel','chipmunk','giraffe','sikadeer','reindeer','weasel','wolf','cat','hyena','guineapig']
imgnum = 0
for key in resnet152_res.keys():
   print "---------------------------------------"
   imgnum = imgnum + 1
   eachimage = key
   prob = (resnet152_res[key] + googlenet_res[key] + resnet101_res[key] + resnet101_testset_res[key] ) / 4
   prob = prob[0]
   top_k = prob.argsort()[-1:-3:-1]
   temp = eachimage.split('.')
   resultfile.write(temp[0])
   for i in np.arange(top_k.size):
        print labelname.index(labels[int(top_k[i])]) , ("%.6f" % prob[int(top_k[i])]) 
        resultfile.write('\t')
        resultfile.write(str(labelname.index(labels[int(top_k[i])])))
        resultfile.write('\t')
        resultfile.write(("%.6f" % prob[int(top_k[i])]))
   resultfile.write('\n')        
# resnet152_file.close()
# resnet101_file.close()
# resultfile.close()
print imgnum