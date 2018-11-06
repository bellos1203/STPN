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
import utils
import os
from train import train
from test import test
from model import StpnModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def main(_):
    # Parsing Arguments
    args = utils.parse_args()
    mode = args.mode
    train_iter = args.training_num
    test_iter = args.test_iter
    ckpt = utils.ckpt_path(args.ckpt)
    input_list = {
        'batch_size': args.batch_size,
        'beta': args.beta,
        'learning_rate': args.learning_rate,
        'ckpt': ckpt,
        'class_threshold': args.class_th,
        'scale': args.scale}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    model = StpnModel()

    # Run Model
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        if mode == 'train':
            sess.run(init)
            train(sess, model, input_list, 'rgb', train_iter)  # Train RGB stream
            sess.run(init)
            train(sess, model, input_list, 'flow', train_iter)  # Train FLOW stream

        elif mode == 'test':
            sess.run(init)
            test(sess, model, init, input_list, test_iter)  # Test


if __name__ == '__main__':
    tf.app.run()
