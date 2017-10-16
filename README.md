# GroupEmotionRecognition
It implements our work in the [EmotiW17 challenge](https://sites.google.com/site/emotiwchallenge/home). 
Details are in paper "Group Emotion Recognition with Individual Facial Emotion CNNs and Global Image based CNNs.ACM International Conference on Multimodal Interaction (ICMI) 2017"
## 0.For the submission
We use facial CNN models and global image based models. Score fusion is used for final results.
To generate our submitted results, just run the shell script "sh run_score_combine.sh"

## 1.Test our models
### 1.Requires
* Caffe, Caffe-matlab
* pytorch 0.20.0 
* pdollar toolbox

### 2.Installation
* unzip LargeMargin_Softmax_Loss-master.zip (Caffe for large margin softmax) and compile it as normal.
* compile matcaffe
* install pytorch 

### 3.Image-based CNN
* cd image_based_cnn/
* mkdir models
* download models from https://drive.google.com/open?id=0B-DiRMXFmUKQX3Qtb25xZkJLS1U
* change pathes in test_models.py to yours
* python test_models.py

### 4.Aligned facial CNN
* cd aligned_facial_cnn
* mkdir models
* download models from https://drive.google.com/open?id=0B-DiRMXFmUKQNTZyUV90MDNXLWM
* change pathes in demo.m to yours
* run demo.m
### 5.Non-aligned facial CNN
* cd aligned_facial_cnn
* change the path for storing cropped faces in mtcnn_detect_crop.m and run
* cd non_aligned_facial_cnn and change pathes in test_emoti17.py
* python test_emoti17.py
