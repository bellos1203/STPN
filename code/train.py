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

import tensorflow as tf
import numpy as np
import utils
import os
import random
from math import ceil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Path & Hyperparameter
NUM_VIDEOS = 200
IMAGE_SIZE = 224
NUM_SEGMENTS = 400
INPUT_PATHS = {'train': {  # THUMOS 14 Validation Set
    'rgb': '../train_data/rgb_features',  # rgb
    'flow': '../train_data/flow_features',  # flow
},
    'test': {  # THUMOS 14 Test Set
        'rgb': '../test_data/rgb_features',  # rgb
        'flow': '../test_data/flow_features',  # flow
    }
}

TRAIN_LABEL_PATH = '../train_data/train_labels.npy'


def train(sess, model, input_list, stream, train_iter):
    batch_size = input_list['batch_size']
    beta = input_list['beta']
    learning_rate = input_list['learning_rate']
    ckpt = input_list['ckpt']

    t_record = os.path.join(ckpt['path'], 'train_record.txt')
    f = open(t_record, 'w')
    saver = tf.train.Saver(max_to_keep=20)

    label = np.load(TRAIN_LABEL_PATH)

    step_per_epoch = ceil(NUM_VIDEOS / batch_size)
    step = 1

    while step <= train_iter:
        shuffle_idx = random.sample(range(1, NUM_VIDEOS + 1), NUM_VIDEOS)
        for mini_step in range(step_per_epoch):
            # Get mini batch index (batch_size)
            minibatch_index = shuffle_idx[mini_step * batch_size: (mini_step + 1) * batch_size]
            minibatch = utils.processVid(minibatch_index, INPUT_PATHS['train'][stream],
                                         NUM_SEGMENTS)  # Return [batch_size, segment nums, 16, 224, 224, 3]
            minibatch_label = label[minibatch_index - np.ones((len(minibatch_index),), dtype=int)].astype(np.float32)
            minibatch_input = minibatch.reshape(len(minibatch_index) * NUM_SEGMENTS, 1024).astype(np.float32)
            sess.run(model.optimizer, feed_dict={model.X: minibatch_input, model.Y: minibatch_label, model.BETA: beta,
                                                 model.LEARNING_RATE: learning_rate})
            step += 1

            # Print the loss and save weights after every 100 iteration
            if step % 100 == 0:
                train_loss = sess.run(model.loss,
                                      feed_dict={model.X: minibatch_input, model.Y: minibatch_label, model.BETA: beta,
                                                 model.LEARNING_RATE: learning_rate})
                print('iter {:d}, {} train loss {:g}'.format(step, stream, train_loss))
                f.write('iter {:d}, {} train loss {:g}\n'.format(step, stream, train_loss))
                f.flush()

                saver.save(sess, os.path.join(ckpt[stream], '{:s}_{:d}'.format(stream, step)))
                print('Checkpoint {:d} Saved'.format(step))
