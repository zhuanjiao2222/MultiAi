"""
Utilize vgg_fcn8 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_fcn import NNet

import tensorflow as tf

import os


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], 'weights',
    #                               "vgg16.npy")
    # vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)
    #
    # vgg_fcn.wd = hypes['wd']
    #
    # vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True)

    # Create network.
    net = NNet.NEWNet_BN({'data': images}, is_training=train, num_classes=2)

    if hypes['arch']['deep_feat'] == "pool5":
        deep_feat = net.layers['conv5_4_k1_bn']
    elif hypes['arch']['deep_feat'] == "fc7":
        deep_feat = net.layers['']
    else:
        raise NotImplementedError

    vgg_dict = {'deep_feat': deep_feat,
                'early_feat': net.layers['sub12_sum/relu']}

    return vgg_dict


# deep_feat [1,12,39,512]  float32
# early_feat [1,48,156,256]
# image [1,384,1248,3]


# fcn8_vgg
# deep_feat [5,12,39,512]  float32
# early_feat [5,48,156,512]
# image [5,384,1248,3]
