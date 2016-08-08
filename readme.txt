An implementation of our CVPR 2016 paper 
"Recurrent Convolutional Network for Video-based Person Re-Identification"

If you use this code please cite:

"Recurrent Convolutional Network for Video-based Person Re-Identification",
N McLaughlin, J Martinez Del Rincon, P Miller, 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

Summary
-------

We perform video re-identification by taking a sequence of images and training a neural network to produce a single feature that represents the whole sequence. The feature vectors for different sequences can be compared using Euclidean distance. A smaller Euclidean distance indicates increased similarity between sequences. The sequence feature is produced using temporal pooling which averages the network's response at all time-steps in the sequence.


Information
-----------

A slightly cleaned up implementation of our video re-id system is provided here. If possible I will clean-up and improve the code in future.

This code is capable of training a video re-identification network on the iLids video or PRID datasets and saving the learned network for later use. The saved network parameters can be loaded from disk and used to run the evaluation code without needing to train the network again.

The optical flow files were produced using the Matlab code in computeOpticalFlow.m 

This matlab code should be used to generate optical flow files before training the neural network. Alternatively, use the flag –dissableOpticalFlow

Note - Modify lines 70-77 of videoReid.lua to point to the directories containing the video-reid datasets and generated optical flow files


Running the code
----------------

For this code to run you must have Torch7 installed with the nn, nnx, cunn, rnn, image, optim and cutorch pacakges.

You must have an Nvidia GPU in order to use CUDA. See http://torch.ch/ for details.

Example command-line options that will allow you to run the code in standard configuration

th videoReid.lua -nEpochs 500 -dataset 1 -dropoutFrac 0.6 -sampleSeqLength 16 -samplingEpochs 100 -seed 1
