## The Code and Experiental results for paper 

#### [Operational Calibration: Debugging Confidence Errors for DNNs in the Field](https://arxiv.org/abs/1910.02352)
This is a folder contrains code and all experimental results in the paper.

#### Datasets
The MNIST/USPS dataset and ImageCLEF dataset are downloaded in this [transfer learning repository](https://github.com/jindongwang/transferlearning).

The Polarity datasets can be download in this [link](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

The CIFAR-10/STL-10 is downloaded by the torchvision, the code for downloading can be found in `/data/data_process.py`.

The CIFAR-100 is downloaded by the torchvision, the code for downloading and mutation can be found in `/data/rotation.py`.

The CIFAR-100 is downloaded by the torchvision, the code for downloading and mutation can be found in `/data/fine_tune.py`.

**The downloaded dataset needs to be placed in folder `/data`**

****

#### Pre-trained models

The code for training model of MNIST/USPS dataset are available in folder `/model`

For ImageCLEF dataset, the pre-trained model are available [here](``https://github.com/jindongwang/transferlearning'').

For CIFAR-100 dataset, the pre-trained model are available in this [pytorch framework](``https://github.com/bearpaw/pytorch-classification'').

The pre-trained model of ImageNet dataset is driven from PyTorch.

**The downloaded dataset needs to be placed in folder `/model`**

****

**I am too lazy to reconstruct my code...** 

**but I think that the key code for this work is very simple (in `GP_build.py`), thus maybe one can easily apply this code...** 

 




