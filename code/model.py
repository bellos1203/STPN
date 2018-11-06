# Copyright 2018 JaeYoo Park
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

import tensorflow as tf

NUM_CLASS = 20
NUM_SEGMENTS = 400


class StpnModel(object):
    def __init__(self):
        # Define inputs
        self.X = tf.placeholder(tf.float32, [None, 1024])  # [Batch * Segment, 1024]
        self.Y = tf.placeholder(tf.float32, [None, NUM_CLASS])  # [Batch, 20]
        self.BETA = tf.placeholder(tf.float32, )
        self.LEARNING_RATE = tf.placeholder(tf.float32, )

        # Define Network
        # 1. Attention Module
        with tf.variable_scope('Attention'):
            attention_layer = tf.layers.dense(self.X, 256, activation=tf.nn.relu)  # Shape = [Batch * Segment, 256]
            self.attention_weights = tf.layers.dense(attention_layer, 1,
                                                     activation=tf.nn.sigmoid)  # Shape = [Batch * Segment, 1]

        # 2. Classification Module
        with tf.variable_scope('Classification'):
            x_vrep_temp = tf.reshape(self.attention_weights * self.X, [-1, NUM_SEGMENTS, 1024],
                                     name="x_vrep_temp")  # [Batch * Segment, 1] .* [Batch * Segment, 1024] = [Batch * Segment, 1024] (broadcst) -> [Batch, Segment, 1024]
            x_vrep = tf.reduce_mean(x_vrep_temp, 1, name="x_vrep")  # [Batch, Segment, 1024] => [Batch, 1024]
            self.class_weight = tf.layers.dense(x_vrep, NUM_CLASS, activation=None,
                                                name='class_weight')  # Use this one for calculate T-CAM  Shape = [Batch, 20]
            self.class_result = tf.sigmoid(self.class_weight)  # Only used at inference time

        # Loss & Optimizer
        c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y,
                                                                        logits=self.class_weight))  # Sigmoid Cross Entropy Loss

        s_loss_temp = tf.reshape(self.attention_weights, [-1, NUM_SEGMENTS, 1])
        s_loss = tf.reduce_mean(tf.norm(s_loss_temp, ord=1, axis=1))

        self.loss = c_loss + self.BETA * s_loss
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
