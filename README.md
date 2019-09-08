# Pose Estimation for Satellites
My contribution to the ESA Kelvin Pose Estimation Challenge. This repository is based on the speed utils (Starter kit for the Kelvins pose estimation competition):

https://gitlab.com/EuropeanSpaceAgency/speed-utils

https://kelvins.esa.int/satellite-pose-estimation-challenge/home/

This approach is based on Convolutional Pose Machines:
https://arxiv.org/abs/1602.00134

The idea is to extract n keypoints from the satellite which are predicted from a convolutional neural network. The point in the 3D room of each of these keypoints is known.
After the neural network identified the keypoints the image is post processed, the 2D keypoints are extracted and together with the known corresponding 3d coordinates within the satellite model the pose can be estimated with PnP (perspective n-point). This resembles a light implementation of this paper (https://arxiv.org/abs/1809.10790) with the core difference that only a single object which is always visible must be identified.
