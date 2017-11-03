# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

caffe_root = os.environ.get('CAFFE_ROOT')

sys.path.insert(0, caffe_root + '/python') #把pycaffe所在路径添加到环境变量
import caffe

#指定网络结构 与 lenet_train_test.prototxt不同
MODEL_FILE = caffe_root + '/examples/mnist/lenet.prototxt'
PRETRAINED = caffe_root + '/examples/mnist/lenet_iter_10000.caffemodel'
#图片已经处理成 lenet.prototxt的输入要求（尺寸28x28）且已经二值化为黑白色
IMAGE_FILE = caffe_root + '/examples/images/number_4.jpg'

input_image = caffe.io.load_image(IMAGE_FILE, color=False)
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
prediction = net.predict([input_image], oversample = False)
caffe.set_mode_cpu()
print 'predicted class:', prediction[0].argmax()
