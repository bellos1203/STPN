STPN - Weakly Supervised Action Localization by Sparse Temporal Pooling Network (reproduced)
============================================================================================
Overview
--------

This repository contains a reproduced code for the paper [__"Weakly Supervised Action Localization by Sparse Temporal Pooling Network"__](https://arxiv.org/abs/1712.05080) by Phuc Nguyen, Ting Liu, Gautam Prasad, and Bohyung Han, __CVPR 2018__.


Usage Guide
===========
* Hardware : TITAN X GPU 

0.Requirements
--------------
* Python3
* Tensorflow 1.6.0
* numpy 1.15.0
* OpenCV 3.4.2
* Sonnet (to extract features from I3D model) 
* Pandas (to evaluate)
* SciPy 1.1.0


1.Preprocessing
---------------
1) Subsample the video with the sampling ratio of 10 frames per second.
2) After sampling the video frames, rescale them to make the smallest dimension of the frame equal to 256 while **preserving the aspect ratio**.
3) Calculate the Optical Flow (TV-L1)
4) Save the rgb frames to `train_data/rgb` and the flow frames to `train_data/flows` with the name of `vid_num/{:06d}.png`. (`test_data/rgb`, `test_data/flows` for the case of test data)
   I simply save the videos as 1,2,3,....200 for the convenience.

* Please refer to the [STPN paper](https://arxiv.org/abs/1712.05080) or the [I3D paper](https://arxiv.org/abs/1705.07750) for more details about preprocessing step.

5) Extract the feature vector of each video by using the code in the "`feature_extraction`" folder. The extracted features will be saved in the `[train/test]_data/[rgb/flow]_features`.
   Since I use the TITAN X GPU which has 12GB Memory, I extract the feature from 16*100 frames which means 100 segments at each time. If you have the GPU with smaller memory, you should extract the feature with the reduced number of segments.
   Please refer to the `extract_feature.sh` in the folder.

2.Train the Model
-----------------
* Run the `train.sh` code. 
* Please refer to the `train.sh` for more details.

3.Test and Extract the Result
-----------------------------
* Run the `test.sh` code. 
* Please refer to the `test.sh` for more details. 
* Note that I excluded two falsely annotated videos, 270, 1496, following the [SSN paper](https://arxiv.org/pdf/1704.06228.pdf).

4.Evaluate
----------
* Run the `eval.sh` code. 
* Please refer to the `eval.sh` for more details. 
* I used the evaluation code from the official [ActivityNet repo](https://github.com/activitynet/ActivityNet), as the authors did.

Reproduced Result
=================
With the provided sample checkpoint(files in the `code/ckpt/ckpt001`), I got the following result for the THUMOS14 test set, which is similar to the paper.

|    tIoU    | 0.1| 0.2| 0.3| 0.4| 0.5| 0.6| 0.7| 0.8| 0.9| mAP|
|------------|----|----|----|----|----|----|----|----|----|----|
| STPN(paper)|52.0|44.7|35.5|25.8|16.9| 9.9| 4.3| 1.2| 0.1|21.2|
| Reproduced |52.1|44.2|34.7|26.1|17.7|10.1| 4.9| 1.3| 0.1|21.3|

Please note that the best result appears around 22k ~ 25k and sometimes the performance could be slightly different from the numbers above.

Comments
--------

If you have any questions or comments, please contact me. <bellos1203@snu.ac.kr>

# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)

License
-------
Apache-2.0
