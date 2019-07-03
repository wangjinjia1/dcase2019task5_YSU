# dcase2019task5_YSU
DCASER2019 task5 urban sound tagging using audio from an urban acoustic sensor network in New York City.

PyTorch implementation

Audio tagging aims to assign one or more labels to the audio clip. In this task, we used the Time-Frequency Segmentation Attention Network (TFSANN) for urban sound tagging. In the training, the log mel spectrogram of the audio clip is used as input feature, and the time-frequency segmentation mask is obtained by the timefrequency segmentation network. The time-frequency segmentation mask can be used to separate the time-frequency domain sound event from the background scene, and enhance the sound event that occurred in the audio clip. Global Weighted Rank Pooling (GWRP) allows existing event categories to occupy significant part of the spectrogram, allowing the network to focus on more significant features, and it can also estimate the probability of existence of sound event. In this paper, the proposed TFSANN model is validated on the development dataset of DCASE2019 task 5. Finally, the coarsegrained and fine-grained taxonomy results are obtained on the Micro Area under precision-recall curve (AUPRC), Micro F1 score and Macro Area under precision-recall curve (AUPRC). 

Requirements:
Python = 3.6
Torch 1.0.1
Librosa 0.6.2

Running step:
(1) download : Downloads the Task 5's data from Zenodo.
(2) extract train, validation and evaluate features:
  python features.py 
(3) train coarse or fine model
python train.py 
(4) inference 
  python inference_validation.py 


Citation
[1] Kong, Qiuqiang, Yong Xu, Iwona Sobieraj, Wenwu Wang, and Mark D. Plumbley. "Sound Event Detection and Time-Frequency Segmentation from Weakly Labelled Data." arXiv preprint arXiv:1804.04715 (2018).
[2] Q. Kong, Y. Xu, I. Sobieraj, W. Wang and M. D. Plumbley, "Sound Event Detection and Time–Frequency Segmentation from Weakly Labelled Data," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 4, pp. 777-787, April 2019.
doi: 10.1109/TASLP.2019.2895254
