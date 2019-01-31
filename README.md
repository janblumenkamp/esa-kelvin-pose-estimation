# speed-utils

Starter kit for the Kelvins pose estimation competition.


### Introduction
The purpose of this repository is to help competitiors of the 
[Kelvins Satellite Pose Estimation Challenge](https://kelvins.esa.int/satellite-pose-estimation-challenge/)
 to get started with working on the SPEED dataset, by providing utility scripts and examples:
  * `visualize_pose.ipynb`: a Jupyter Notebook for inspecting the dataset: it plots example images,
  pose label is visualized by projected axes.
  * `submission.py`: utility class for generating valid submissions.
  * `fake_submission_gen.py`: submission generation example.
  * `pytorch_example.py` and `keras_example.py`: training on SPEED with Keras and Pytorch deep learning
  frameworks.
  * `utils.py`: utility scripts for the above examples (projection to camera frame, Keras DataGenerator
  for SPEED, PyTorch Dataset, etc.). 
  
### Setting up
Clone this repository:
```
git clone https://gitlab.com/EuropeanSpaceAgency/speed-utils.git
cd speed-utils
```
Install dependencies:  
```
pip install numpy pillow matplotlib
pip install torch torchvision  # optional for running pytorch example
pip install tensorflow-gpu  # optional for running keras example
pip install jupyter  # optional for running notebook
```

### Training examples

We provide example training scripts for two popular Deep Learning frameworks: for Keras and PyTorch.
These examples are intentionally kept simple, leaving lots of room for improvement (dataset augmentation,
more suitable loss functions, normalizing outputs, exploring different network architectures, and so on).

Starting PyTorch training:

```
python pytorch_example.py --dataset [path to downloaded dataset] --epochs [num epochs] --batch [batch size]
```
 

Similarly, to start Keras training:

```
python keras_example.py --dataset [path to downloaded dataset] --epochs [num epochs] --batch [batch size]
```

As the training is finished, the model is evaluated on all images of the `training` and `real_training`
sets, and a submission file is generated that can be directly submitted on the competition page.