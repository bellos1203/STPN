# Copyright 2018 Jae Yoo Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Functions for extract_feature.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import i3d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

IMAGE_SIZE = 224
NUM_CLASSES = 400

SAMPLE_VIDEO_FRAMES = 16

CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow': 'data/checkpoints/flow_imagenet/model.ckpt',
}

LABEL_MAP_PATH = 'data/label_map.txt'


def get_model(streamType, numSeg):
    if streamType == 'rgb':
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
           tf.float32,
           shape=(numSeg, SAMPLE_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
               NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
               rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    elif streamType == 'flow':
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
           tf.float32,
           shape=(numSeg, SAMPLE_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
               NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
               flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if streamType == 'rgb':
        model_logits = rgb_logits
        inputs = rgb_input
    elif streamType == 'flow':
        model_logits = flow_logits
        inputs = flow_input
    
    return saver, inputs, model_logits


def get_feature(X, streamType, numSeg, fSaver, fInput, mLogits):
    eval_type = streamType
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
  
    with tf.Session(config=config) as sess:
        X_features = np.zeros((numSeg, 1024))
        feed_dict = {}
        if eval_type in ['rgb', 'joint']:
            rgb_input = fInput
            fSaver.restore(sess, CHECKPOINT_PATHS['rgb'])
            feed_dict[rgb_input] = X

        if eval_type in ['flow', 'joint']:
            flow_input = fInput
            fSaver.restore(sess, CHECKPOINT_PATHS['flow'])
            feed_dict[flow_input] = X

        out_logits = sess.run(mLogits, feed_dict=feed_dict)
        out_logits = np.squeeze(out_logits)
        print(out_logits.shape)
        X_features = out_logits
    return X_features

