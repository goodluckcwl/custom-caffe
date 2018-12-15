import sys
sys.path.insert(0, '/home/chenweiliang/caffe-windows-ms/python')
import caffe

import numpy as np
from PIL import Image
import scipy.io as sio
import os

import random

class LandmarkVisualLayer(caffe.Layer):
    """
    Save images to file
    """
    def setup(self, bottom, top):
        pass
        """
        - save_dir: path to save images
        - interval
        """
        # config
        params = eval(self.param_str)
        self.save_dir = params['save_dir']
        self.linesize = int(params['linesize'])
        self.iter = 0
        print self.save_dir
        # 1 top
        if len(top) != 1:
            raise Exception("Need to define one top.")
        # data layers have no bottoms
        if len(bottom) != 2:
            raise Exception("Need to define two bottom:data,label.")

        # Make output_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass
        # assign output
        self.data = top[0].data[...]
        self.label = top[1].data[...]
        self.iter = self.iter + 1

        #
        n_images = self.data.shape[0]
        channels = self.data.shape[1]
        height = self.data.shape[2]
        width = self.data.shape[3]

        # save images
        self.label[0:68] = self.label[0:68] * width - 1
        self.label[68:] = self.label[68:] * height - 1
        idx = random.randint(0, n_images)
        image = self.data[idx] / 0.0078125 + 127.5
        transformer = caffe.io.Transformer({'data': [1, 3, height, width]})
        transformer.set_transpose('data', (1, 2, 0))  # c,h,w-> h,w,c
        img = transformer.preprocess(image)
        # pad to be large enough
        pad = int(width * 0.3)
        label = label + pad
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad)
        for x,y in zip(label[0:68], label[68:]):
            x = int(x)
            y = int(y)
            x_begin = x - self.linesize
            x_end = x + self.linesize
            y_begin = y - self.linesize
            y_end = y + self.linesize
            for j in range(y_begin, y_end+1, 1):
                for i in range(x_begin, x_end+1, 1):
                    if i < img_pad.shape[1] and i >= 0 and j < img_pad.shape[0] and j >= 0:
                        img_pad[j][i][0] = 0
                        img_pad[j][i][1] = 255
                        img_pad[j][i][2] = 0
        # Save
        cv2.imwrite(self.save_dir + '/%d.jpg'%self.iter, img_pad.astype(np.uint8))






    def backward(self, top, propagate_down, bottom):
        pass



class FaceDataLayer(caffe.Layer):
    """
     Load (input image, label image) pairs from 300W
     one-at-a-time while reshaping the net to preserve dimensions.
    
     Use this to feed data to a fully convolutional network.
     """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
    
        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
    
        for PASCAL VOC semantic segmentation.
    
        example
    
        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.root_dir = params['root_dir']
        self.file_dir = params['file_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.scale = params['scale']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        fcn_stage1 = sio.loadmat(self.file_dir)['fcn_stage1'][0, 0]
        self.landmarks = fcn_stage1['landmarks']
        self.imglist = fcn_stage1['imglist']
        self.indices = np.arange(1,len(self.imglist),1)
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}'.format(self.root_dir, self.imglist[idx][0][0]))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ *= self.scale
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = sio.loadmat('{}/{}'.format(self.root_dir, self.imglist[idx][0][0][:-4] + '.mat'))['heatmaps']
        #label = label[np.newaxis, ...]
        return label