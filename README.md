# NormProjection
This project is the Torch implementation of the paper: Projection Based Weight Normalization for Deep Neural Networks (arXiv:1710.02338)

* bibtex:
```Bash
@article{Huang_2017_arxiv,
    author = {Lei Huang and Xianglong Liu and  Bo Lang  and Bo Li},
    title = {:Projection Based Weight Normalization for Deep Neural Networks},
   journal   = {CoRR},
  volume    = {abs/1710.02338},
  year      = {2017}}
 ```
 
## Requirements and Dependency
* Install [Torch](http://torch.ch) with CUDA GPU
* Install [cudnn v5](http://torch.ch)
* Install dependent lua packages optnet by run:
luarocks install optnet


## Experiments in the paper
	
#### 1. Reproduce the results on MLP architecture over MNIST dataset:

* Execute:
```Bash
  bash 1_execute_MLP.sh   
  bash 1_execute_MLP_UpdateT.sh
 ```

 
#### 2. Reproduce the results on Incption, VGG and Residual network over CIFAR datsets: 

 *	Dataset preparations: you should download the [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) and [CIFAR-100](https://yadi.sk/d/ZbiXAegjxaBcM) datasets, and put the data file in the directory: './dataset/' 

  *	To reproduce the experimental results, you can run the script below, which include all the information of experimental configurations: 
```Bash
  bash 2_execute_Conv_Inception.sh  
  bash 3_execute_Conv_VGG.sh 
  bash 4_execute_Conv_resnet.sh  
 ```
 The Inception model is based on the project on: https://github.com/soumith/imagenet-multiGPU.torch.
 
 The residual network  model is based on the facebook torch project: https://github.com/facebook/fb.resnet.torch


#### 3. Run the experiment on imageNet dataset. 

 *  (1) You should clone the facebook residual network project from:https://github.com/facebook/fb.resnet.torch
 *  (2) You should download imageNet dataset and put it on: '/tmp/dataset/imageNet/' directory (you also can use other path, which can be set in 'opts_imageNet.lua')
 *  (3) Copy  'opts_imageNet.lua', 'exp_Conv_imageNet_expDecay.lua', 'train_expDecay.lua', 'module' and 'models' to the project's root path.
 *  (4)	Execute: 
```Bash
th exp_Conv_imageNet_expDecay.lua -model imagenet/preresnet_BN -LR 0.05
 ```
You can training other respective models by using the parameter '-model'


#### 4. Semi-supervised learning experiments on Ladder networks
The semi-supervised tasks based on Ladder network can be find in this project: https://github.com/huangleiBuaa/Ladder_deepSSL_NP


## Contact
huanglei@nlsde.buaa.edu.cn, Any discussions and suggestions are welcome!

