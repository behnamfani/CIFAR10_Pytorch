# CIFAR10_Pytorch
Creating a Convolutional Neural Network for CIFAR10 [1] dataset

In this project, I built a CNN to learn and predict CIFAR10. 

Libraries I am using are:

<pre>
-import torch
-import torchvision
-import torch.nn as nn
-import torch.nn.functional as F
-import torch.optim as optim
-import torchvision.transforms as transforms
-import matplotlib.pyplot as plt 
-import matplotlib.pyplot as plt
-from matplotlib import style
-import numpy as np
-import cv2
-from tqdm import tqdm
</pre>

We can run this model on GPU as well:

if torch.cuda.is_available():
  print('Run on GPU')
  device = torch.device("cuda:0")
else:
  print('Run on CPU')
  device = torch.device("cpu")
  
First, the program downloads the CIFAR10 dataset and creates a trainset and a testset.

Then, this is the model that the program creates, trains, and predicts with:

<pre>
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=2048, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</pre>

These are the history of accuracies and losses of the model running on trainset during training:

![](reports/acc.png)

And Finally, this is model accuracy and loss for testset:

Test_Accuracy = 0.66, Test_Loss = 0.9756

[1]: https://www.cs.toronto.edu/~kriz/cifar.html
