# ConvFNN
The code for the paper "Convolutional Fuzzy Neural Networks With Random Weights for Image Classification".

**Reference:**    
Y. Wang, H. Ishibuchi, W. Pedrycz, J. Zhu, X. Cao and J. Wang, "Convolutional Fuzzy Neural Networks With Random Weights for Image Classification," *IEEE Transactions on Emerging Topics in Computational Intelligence*, vol. 8, no. 5, pp. 3279-3293, 2024.

@ARTICLE{10476629,  
  author={Wang, Yifan and Ishibuchi, Hisao and Pedrycz, Witold and Zhu, Jihua and Cao, Xiangyong and Wang, Jun},   
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},    
  title={Convolutional Fuzzy Neural Networks With Random Weights for Image Classification},    
  year={2024},   
  volume={8},   
  number={5},   
  pages={3279-3293},   
}

This paper can be downloaded from https://ieeexplore.ieee.org/abstract/document/10476629.

## Introduction:
Deep fuzzy neural networks have established a fundamental connection between fuzzy systems and deep learning networks, serving as a crucial bridge between two research fields in computational intelligence. These hybrid networks have powerful learning capability stemming from deep neural networks while leveraging the advantages of fuzzy systems, such as robustness. Due to these benefits, deep fuzzy neural networks have recently been an emerging topic in computational intelligence. With the help of deep learning, fuzzy systems have achieved great performance on the classification task. Although fuzzy systems have been extensively investigated, they still struggle in terms of image classification. In this paper, we propose a convolutional fuzzy neural network that combines improved convolutional neural networks with a fuzzy-set-based fusion technique. Different from convolutional neural networks, filters are randomly generated in convolutional layers in our model. This operation not only leads to the fast learning of the model but also avoids some notorious problems of gradient descent procedures in conventional deep learning methods.

## Structure:
<img width="1453" height="774" alt="image" src="https://github.com/user-attachments/assets/66bcb337-876d-43ac-980d-894032e79fef" />

The main structure of ConvFNN includes an input layer, a fuzzy image layer, convolutional layers, pooling layers, and an output layer. The input layer receives images as
the input data to the entire pipeline. Then, the fuzzy image layer generates several fuzzy images, each of which corresponds to one fuzzy rule. The convolutional and pooling layers subsequently perform convolution and pooling operations on the fuzzy images, respectively. Several convolutional and pooling layers can be stacked to extract effective features for classification. Finally, the output layer predicts the class label of the input pattern based on these high-level features.

## Experimental Results:   
The experimental results of ConvFNN are shown as follows:   

| Datasets | Accuracy |
|:-------|:--------|
|COIL20|100%|
|COIL100|91.38%|
|JAFFE|100%|
|ORL|100%|
|Leaves|95.92%|
|MPEG7|91.95%|
|CASIA-faceV5|91%|
|UMIST|89.94%|
|Yale|100%|
|USPS|98.4%|

## How to run the code:   
### Prepare your datasets: 
| DataSets | Dimension |
|:-------|:--------|
|x_train|N_trainxD|
|x_test|N_testxD|
|y_train|N_train|
|y_test|N_test|

Here, $x$ is the features and $y$ is the lables. $N$ is the number of training/testing patterns and $D$ is the dimension of features.

### Use the folowing command in the MatLab:  
```
clear
clc
addpath(genpath('functions'));
load data.mat %Here is your data
load param.mat
test_accuracy = ConvFNN(param, x_train, x_test, y_train, y_test);
```

Thanks for your attention.


