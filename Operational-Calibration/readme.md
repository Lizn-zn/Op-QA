## The Code and Experiental results for paper 

#### [Operational Calibration: Debugging Confidence Errors for DNNs in the Field](https://arxiv.org/abs/1910.02352)
This is a folder contrains code and all experimental results in the paper.

#### Datasets
The MNIST/USPS dataset and ImageCLEF dataset are downloaded in this [transfer learning repository](https://github.com/jindongwang/transferlearning).

The Polarity datasets can be download in this [link](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

The CIFAR-10, STL-10, CIFAR-100, and ImageNet are downloaded by the torchvision, the code for downloading can be found in `dataset/data`.

For mutation operations in the other two datasets, the code can be found in `dataset/data`

#### pre-trained models
For ImageCLEF dataset, the pre-trained model are available [here](``https://github.com/jindongwang/transferlearning'').
For CIFAR-100 dataset, the pre-trained model are available in this [pytorch framework](``https://github.com/bearpaw/pytorch-classification'').
The pre-trained model of ImageNet dataset is driven from PyTorch.

#### Experimental results

- The Efficacy of our calibration method are evaluated in `/Efficacy`.
- The Efficiency of our calibration method are evaluated in `/Efficiency`.
- The Surprise Adequacy Regression can be found in `/SA-exp`.
- The other two machine learning model for regression can be found in `/Regression`.
- The temperature scaling is used from this [repository](https://github.com/gpleiss/temperature_scaling), and the code can be found in `/Temperature-scaling`.




