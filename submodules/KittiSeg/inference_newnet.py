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

    logits = {}

    logits['images'] = images

    if hypes['arch']['fcn_in'] == 'pool5':
        logits['fcn_in'] = net.layers['']
    elif hypes['arch']['fcn_in'] == 'fc7':
        logits['fcn_in'] = net.layers['conv5_3_sum']
    else:
        raise NotImplementedError

    logits['feed2'] = net.layers["sub24_sum/relu"]
    logits['feed4'] = net.layers["sub12_sum/relu"]

    # logits['feed4'] = vgg_fcn.pool3

    logits['fcn_logits'] = net.layers["conv6_cls_out"]

    return logits

# fcn_in [1,12,39,2048]  float32
# fcn_logits [1,384,1248,2]
# feed2 [1,24,78,256]
# feed4 [1,48,156,256]
# image [1,384,1248,3]

# fcn8_vgg
# fcn_in [1,?,?,4096]  float32
# fcn_logits [?,?,?,2]
# feed2 [1,?,?512]
# feed4 [1,?,?256]
# image [1,?,?,3]
