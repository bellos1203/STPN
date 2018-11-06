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

import numpy as np
import tensorflow as tf
import os
import utils_feature as uf
import argparse
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

IMAGE_SIZE = 224
INPUT_VIDEO_FRAMES = 16
NUM_VIDEOS = {'train': 200, 'test': 210}
INPUT_PATHS = {'train': {'rgb': '../train_data/rgb',  # rgb
                         'flow': '../train_data/flows',  # flow
                         },
               'test': {'rgb': '../test_data/rgb',  # rgb
                        'flow': '../test_data/flows',  # flow
                        }
               }
SAVE_PATHS = {'train': {'rgb': '../train_data/rgb_features',  # rgb
                        'flow': '../train_data/flow_features',  # flow
                        },
              'test': {'rgb': '../test_data/rgb_features',  # rgb
                       'flow': '../test_data/flow_features',  # flow
                       }
              }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default='rgb', type=str)
    parser.add_argument('--data', default='train', type=str)
    argms = parser.parse_args()
    return argms


# Main : extract features
args = parse_args()
stream = args.stream
data = args.data

init = tf.global_variables_initializer()

for i in range(1, NUM_VIDEOS[data] + 1):
    vid_path = os.path.join(INPUT_PATHS[data][stream], '{:d}'.format(i))  # path
    num_vid_frame = len(os.listdir(vid_path))  # number of total frames
    num_segments = int(num_vid_frame / INPUT_VIDEO_FRAMES)  # number of total segments (ex. 1 feature vector per 16 frames, so total segments will be (total frames / 16))
    # Load feature model
    print('model {:d} loaded'.format(i))
    print('{:d} : {:d} frames, {:d} segs'.format(i, num_vid_frame, num_segments))
    X_feature = np.zeros((num_segments, 1024))
    num_hundred_segments = int(num_segments / 100)
    # Load Images
    channel = 3
    vid = np.zeros((num_vid_frame, IMAGE_SIZE, IMAGE_SIZE, channel))
    for frame in range(num_vid_frame):
        frm = os.path.join(vid_path, '{:06d}.png'.format(frame))
        vid[frame] = cv2.imread(frm, cv2.IMREAD_COLOR)
    print('{:d} vid frames loaded'.format(i))

    # Feature Extracting
    vid = vid.astype(np.float32)
    vid = 2.0 * (vid / 255.0) - 1.0
    if stream == 'rgb':
        vid = vid[:, :, :, ::-1]  # convert BGR to RGB
    if stream == 'flow':
        channel = 2
        vid = vid[:, :, :, 0:2]
    for j in range(num_hundred_segments + 1):
        if j == num_hundred_segments:
            extract_size = num_segments - num_hundred_segments * 100
        else:
            extract_size = 100

        tf.reset_default_graph()
        feature_saver, feature_input, model_logits = uf.get_model(stream, extract_size)
        frame_inputs = np.zeros((extract_size, INPUT_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, channel))
        for k in range(extract_size):
            frame_inputs[k] = vid[j * INPUT_VIDEO_FRAMES * 100 + k * INPUT_VIDEO_FRAMES:j * INPUT_VIDEO_FRAMES * 100 + (k + 1) * INPUT_VIDEO_FRAMES]
        X_feature[j * 100: j * 100 + extract_size] = uf.get_feature(frame_inputs, stream, extract_size, feature_saver, feature_input, model_logits)

    # Save X_feature
    print('{:d} feature extracted'.format(i))
    npName = os.path.join(SAVE_PATHS[data][stream], '{:d}.npy'.format(i))
    np.save(npName, X_feature)
    print('{:d} feature saved'.format(i))
