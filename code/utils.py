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

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
from scipy.interpolate import interp1d

NUM_SEGMENTS = 400
SAMPLING_FRAMES = 10
NUM_INPUT_FRAMES = 16
NUM_CLASS = 20

CLASS = {0: 'BaseballPitch', 1: 'BasketballDunk', 2: 'Billiards', 3: 'CleanAndJerk', 4: 'CliffDiving', 5: 'CricketBowling', 6: 'CricketShot', 7: 'Diving', 8: 'FrisbeeCatch', 9: 'GolfSwing',
         10: 'HammerThrow', 11: 'HighJump', 12: 'JavelinThrow', 13: 'LongJump', 14: 'PoleVault', 15: 'Shotput', 16: 'SoccerPenalty', 17: 'TennisSwing', 18: 'ThrowDiscus', 19: 'VolleyballSpiking'}

CLASS_INDEX = {'BaseballPitch': '7', 'BasketballDunk': '9', 'Billiards': '12', 'CleanAndJerk': '21', 'CliffDiving': '22', 'CricketBowling': '23', 'CricketShot': '24', 'Diving': '26', 'FrisbeeCatch': '31', 'GolfSwing': '33',
         'HammerThrow': '36', 'HighJump': '40', 'JavelinThrow': '45', 'LongJump': '51', 'PoleVault': '68', 'Shotput': '79', 'SoccerPenalty': '85', 'TennisSwing': '92', 'ThrowDiscus': '93', 'VolleyballSpiking': '97'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str)
    # input for training
    parser.add_argument('--training_num', default=80000, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--beta', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--ckpt', type=int)
    # input for inference
    parser.add_argument('--test_iter', default=0, type=int)
    parser.add_argument('--class_th', default=0.1, type=float)
    parser.add_argument('--scale', default=24, type=int)
    args = parser.parse_args()
    return args


def ckpt_path(c):
    directory = os.path.join('ckpt', 'ckpt{:03d}'.format(c))
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, 'rgb'))
        os.makedirs(os.path.join(directory, 'flow'))

    cp = dict(path=directory, rgb=os.path.join(directory, 'rgb'),
              flow=os.path.join(directory, 'flow'))
    return cp


def inf_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def random_perturb(v_len):
    random_p = np.arange(NUM_SEGMENTS) * v_len / NUM_SEGMENTS
    for i in range(NUM_SEGMENTS):
        if i < NUM_SEGMENTS - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(vid_len):
    u_sample = np.arange(NUM_SEGMENTS) * vid_len / NUM_SEGMENTS
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)


# Process training data
def processVid(idx, f_path, numSeg):
    batch_frames = np.zeros((len(idx), numSeg, 1024))
    for i in range(len(idx)):
        numvid = idx[i]
        feature_path = os.path.join(f_path, '{:d}.npy'.format(numvid))
        feature = np.load(feature_path).astype(np.float32)
        seg_list = random_perturb(feature.shape[0])
        batch_frames[i] = feature[seg_list]

    return batch_frames


# Process test data
def processTestVid(idx, fpath, numSeg):
    rgb_frames = np.zeros((numSeg, 1024))
    flow_frames = np.zeros((numSeg, 1024))

    rgbpath = os.path.join(fpath['rgb'], '{:d}.npy'.format(idx))
    flowpath = os.path.join(fpath['flow'], '{:d}.npy'.format(idx))

    rvid = np.load(rgbpath).astype(np.float32)
    fvid = np.load(flowpath).astype(np.float32)

    seg_list = uniform_sampling(rvid.shape[0])

    rgb_frames = rvid[seg_list]
    flow_frames = fvid[seg_list]

    return rgb_frames, flow_frames, seg_list, rvid.shape[0]


# Localization functions (post-processing)
# Get TCAM signal
def get_tCAM(feature, layer_Weights):
    tCAM = np.matmul(feature, layer_Weights)
    return tCAM


# Get weighted TCAM and the score for each segment
def get_wtCAM(main_tCAM, sub_tCAM, attention_Weights, alpha, pred):
    wtCAM = attention_Weights * tf.nn.sigmoid(main_tCAM)
    signal = np.reshape(wtCAM.eval()[:, pred], (NUM_SEGMENTS, -1, 1))
    score = np.reshape((attention_Weights * (alpha * main_tCAM + (1 - alpha) * sub_tCAM))[:, pred],
                       (NUM_SEGMENTS, -1, 1))
    return np.concatenate((signal, score), axis=2)


# Interpolate empty segments
def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')  # linear/quadratic/cubic
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


# Interpolate the wtCAM signals and threshold
def interpolated_wtCAM(wT, scale):
    final_wT = upgrade_resolution(wT, scale)
    result_zero = np.where(final_wT[:, :, 0] < 0.05)
    final_wT[result_zero] = 0
    return final_wT


# Return the index where the wtcam value > 0.05
def get_tempseg_list(wtcam, c_len):
    temp = []
    for i in range(c_len):
        pos = np.where(wtcam[:, i, 0] > 0)
        temp_list = pos
        temp.append(temp_list)
    return temp


# Group the connected results
def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


# Get the temporal proposal
def get_temp_proposal(tList, wtcam, c_pred, scale, v_len):
    t_factor = (NUM_INPUT_FRAMES * v_len) / (scale * NUM_SEGMENTS * SAMPLING_FRAMES)  # Factor to convert segment index to actual timestamp
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)  # Get the connected parts
            for j in range(len(grouped_temp_list)):
                c_score = np.mean(wtcam[grouped_temp_list[j], i, 1])
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])  # Add the proposal
        temp.append(c_temp)
    return temp


# Perform Non-Maximum-Suppression
def nms_prop(arr):
    p = [2, 0, 1]
    idx = np.argsort(p)
    prop_tensor = (arr[:, 1:])[:, idx]
    fake_y = np.tile(np.array([0, 1]), (arr.shape[0], 1))
    box = prop_tensor[:, :2]
    score = prop_tensor[:, 2]
    box_prop = np.concatenate((fake_y, box), 1)
    p2 = [0, 2, 1, 3]
    pidx = np.argsort(p2)
    box_prop = box_prop[:, pidx]
    result = tf.image.non_max_suppression(box_prop, score, max_output_size=1000, iou_threshold=0.5)
    return result.eval()


#  Fuse two stream & perform non-maximum suppression
def integrated_prop(rgbProp, flowProp, rPred, fPred):
    temp = []
    for i in range(NUM_CLASS):
        if (i in rPred) and (i in fPred):
            ridx = rPred.index(i)
            fidx = fPred.index(i)
            rgb_temp = rgbProp[ridx]
            flow_temp = flowProp[fidx]
            rgb_set = set([tuple(x) for x in rgb_temp])
            flow_set = set([tuple(x) for x in flow_temp])
            fuse_temp = np.array([x for x in rgb_set | flow_set])  # Gather RGB proposals and FLOW proposals together
            fuse_temp = np.sort(fuse_temp.view('f8,f8,f8,f8'), order=['f1'], axis=0).view(np.float)[::-1]

            if len(fuse_temp) > 0:
                nms_idx = nms_prop(fuse_temp)
                for k in nms_idx:
                    temp.append(fuse_temp[k])

        elif (i in rPred) and not (i in fPred):  # For the video which only has RGB Proposals
            ridx = rPred.index(i)
            rgb_temp = rgbProp[ridx]
            for j in range(len(rgb_temp)):
                temp.append(rgb_temp[j])
        elif not (i in rPred) and (i in fPred):  # For the video which only has FLOW Proposals
            fidx = fPred.index(i)
            flow_temp = flowProp[fidx]
            for j in range(len(flow_temp)):
                temp.append(flow_temp[j])
    return temp


# Record the proposals to the json file
def result2json(result):
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': CLASS[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file


def json2txt(jF, rF):
    for i in jF.keys():
        for j in range(len(jF[i])):
            rF.write('{:s} {:f} {:f} {:s} {:f}\n'.format(i, jF[i][j]['segment'][0], jF[i][j]['segment'][1],
                                                        CLASS_INDEX[jF[i][j]['label']], round(jF[i][j]['score'], 6)))


