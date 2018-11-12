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
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Path & Hyperparameter
NUM_SEGMENTS = 400  # Should be larger : 400
INPUT_PATHS = {'train': {  # THUMOS 14 Validation Set
    'rgb': '../train_data/rgb_features',  # rgb
    'flow': '../train_data/flow_features',  # flow
},
    'test': {  # THUMOS 14 Test Set
        'rgb': '../test_data/rgb_features',  # rgb
        'flow': '../test_data/flow_features',  # flow
    },
}
TEST_NUM = 210  # I excluded two falsely annotated videos, 270, 1496, following the SSN paper (https://arxiv.org/pdf/1704.06228.pdf)
ALPHA = 0.5


def test(sess, model, init, input_list, test_iter):
    ckpt = input_list['ckpt']
    scale = input_list['scale']
    class_threshold = input_list['class_threshold']

    rgb_saver = tf.train.Saver()
    flow_saver = tf.train.Saver()
    test_vid_list = open('THUMOS14_test_vid_list.txt', 'r')  # file for matching 'video number' and 'video name'
    lines = test_vid_list.read().splitlines()

    # Define json File (output)
    final_result = {}
    final_result['version'] = 'VERSION 1.3'
    final_result['results'] = {}
    final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    for i in range(1, TEST_NUM + 1):
        vid_name = lines[i - 1]
        # Load Frames
        rgb_features, flow_features, temp_seg, vid_len = utils.processTestVid(i, INPUT_PATHS['test'],
                                                                              NUM_SEGMENTS)
        rgb_features = rgb_features.astype(np.float32)
        flow_features = flow_features.astype(np.float32)

        # RGB Stream
        sess.run(init)
        rgb_saver.restore(sess, os.path.join(ckpt['rgb'], 'rgb_' + str(test_iter)))
        rgb_class_w = tf.get_default_graph().get_tensor_by_name('Classification/class_weight/kernel:0').eval()
        rgb_attention, rgb_raw, rgb_class_result = sess.run(
            [model.attention_weights, model.class_weight, model.class_result],
            feed_dict={model.X: rgb_features})

        # Flow Stream
        sess.run(init)
        flow_saver.restore(sess, os.path.join(ckpt['flow'], 'flow_' + str(test_iter)))
        flow_class_w = tf.get_default_graph().get_tensor_by_name('Classification/class_weight/kernel:0').eval()
        flow_attention, flow_raw, flow_class_result = sess.run(
            [model.attention_weights, model.class_weight, model.class_result],
            feed_dict={model.X: flow_features})

        # Gathering Classification Result
        rgb_class_prediction = np.where(rgb_class_result > class_threshold)[1]
        flow_class_prediction = np.where(flow_class_result > class_threshold)[1]

        rgb_tCam = utils.get_tCAM(rgb_features, rgb_class_w)
        flow_tCam = utils.get_tCAM(flow_features, flow_class_w)
        r_check = False
        f_check = False

        if rgb_class_prediction.any():
            r_check = True
            # Weighted T-CAM
            rgb_wtCam = utils.get_wtCAM(rgb_tCam, flow_tCam, rgb_attention, ALPHA, rgb_class_prediction)
            # Interpolate W-TCAM
            rgb_int_wtCam = utils.interpolated_wtCAM(rgb_wtCam, scale)
            # Get segment list of rgb_int_wtCam
            rgb_temp_idx = utils.get_tempseg_list(rgb_int_wtCam, len(rgb_class_prediction))
            # Temporal Proposal
            rgb_temp_prop = utils.get_temp_proposal(rgb_temp_idx, rgb_int_wtCam, rgb_class_prediction,
                                                    scale, vid_len)

        if flow_class_prediction.any():
            f_check = True
            # Weighted T-CAM
            flow_wtCam = utils.get_wtCAM(flow_tCam, rgb_tCam, flow_attention, 1 - ALPHA, flow_class_prediction)
            # Interpolate W-TCAM
            flow_int_wtCam = utils.interpolated_wtCAM(flow_wtCam, scale)
            # Get segment list of flow_int_wtCam
            flow_temp_idx = utils.get_tempseg_list(flow_int_wtCam, len(flow_class_prediction))
            # Temporal Proposal
            flow_temp_prop = utils.get_temp_proposal(flow_temp_idx, flow_int_wtCam, flow_class_prediction,
                                                     scale, vid_len)

        if r_check and f_check:
            # Fuse two stream and perform non-maximum suppression
            temp_prop = utils.integrated_prop(rgb_temp_prop, flow_temp_prop, list(rgb_class_prediction),
                                              list(flow_class_prediction))
            final_result['results'][vid_name] = utils.result2json([temp_prop])
        elif r_check and not f_check:
            final_result['results'][vid_name] = utils.result2json(rgb_temp_prop)
        elif not r_check and f_check:
            final_result['results'][vid_name] = utils.result2json(flow_temp_prop)

        utils.inf_progress(i, TEST_NUM, 'Progress', 'Complete', 1, 50)

    # Save Results
    json_path = os.path.join(ckpt['path'], 'results.json')
    with open(json_path, 'w') as fp:
        json.dump(final_result, fp)


    txt_path = os.path.join(ckpt['path'], 'results.txt')
    with open(txt_path, 'w') as tp:
        utils.json2txt(final_result['results'], tp)


    test_vid_list.close()
