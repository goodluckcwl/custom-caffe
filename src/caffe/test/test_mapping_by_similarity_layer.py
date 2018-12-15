# -*- coding: utf-8 -*-
# @Author: chenweiliang
# @Date:   2018-05-02 16:12:13
# @Last Modified by:   chenweiliang
# @Last Modified time: 2018-05-02 16:13:16

import sys
sys.path.insert(0, '/home/chenweiliang/caffe-windows-ms/python')
import caffe
import numpy as np

# Test cpu
# caffe.set_mode_cpu()
# Test gpu
caffe.set_mode_gpu()
net = caffe.Net('mapping_by_similarity_layer.prototxt', caffe.TEST)
random_input = (np.random.random([8, 64, 16, 16]) - 0.5)*2
# random_input = np.ones([2, 64, 16, 16])
n_images, channels, height, width = random_input.shape

# Normalize
random_norm_input = np.zeros(random_input.shape)
for n in range(n_images):
    for h in range(height):
        for w in range(width):
            l2norm_data = np.linalg.norm(random_input[n, :, h, w])
            random_norm_input[n, :, h, w] = random_input[n, :, h, w]/l2norm_data

# Similarity Map input. 0~1
sim_input = np.random.random([4, 16, 16])
random_norm_input[0:4, 63, :, :] = sim_input

epsilon = 1e-4
thresh = 1e-3
grads_numeric = np.zeros(random_input.shape)
for n in range(n_images):
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                input_minus = random_norm_input.copy()
                input_minus[n, c, h, w] = input_minus[n, c, h, w] - epsilon
                net.blobs['input'].data[...] = input_minus
                net.forward()
                output1 = net.blobs['mapping'].data.copy()

                input_plus = random_norm_input.copy()
                input_plus[n, c, h, w] = input_plus[n, c, h, w] + epsilon
                net.blobs['input'].data[...] = input_plus
                net.forward()
                output2 = net.blobs['mapping'].data.copy()
                delta = output2 - output1
                if c==63 and n<=3:
                    v1 = sum(sum(sum(sum(delta[0:8:2, :, :, :]))))/2/epsilon + sum(sum(sum(sum(delta[1:8:2, :,:,:]))))/2/epsilon
                v = sum(sum(sum(sum(delta))))/2/epsilon
                # print v
                grads_numeric[n, c, h, w] = v

net.blobs['mapping'].diff[...] = np.ones(output1.shape)
net.blobs['input'].data[...] = random_norm_input
net.forward()
grads_cal = net.backward()['input']
grads_res = abs(grads_cal - grads_numeric)

print('==========>Gradients Check...')
print('max difference between numeric gradients and calculated gradients:%.6f' % grads_res.max())
if grads_res.max() > thresh:
    print('==========>Gradients Check Fail.')
else:
    print('==========>Gradients Check Sucess!')
