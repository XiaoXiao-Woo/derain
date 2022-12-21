# DID-MDN
## Density-aware Single Image De-raining using a Multi-stream Dense Network
[He Zhang](https://sites.google.com/site/hezhangsprinter), [Vishal M. Patel](http://www.rci.rutgers.edu/~vmp93/)

[[Paper Link](https://arxiv.org/abs/1802.07412)] (CVPR'18)

We present a novel density-aware multi-stream densely connected convolutional neural
network-based algorithm, called DID-MDN, for joint rain density estimation and de-raining. The proposed method
enables the network itself to automatically determine the rain-density information and then efficiently remove the
corresponding rain-streaks guided by the estimated rain-density label. To better characterize rain-streaks with dif-
ferent scales and shapes, a multi-stream densely connected de-raining network is proposed which efficiently leverages
features from different scales. Furthermore, a new dataset containing images with rain-density labels is created and
used to train the proposed density-aware network. 

	@inproceedings{derain_zhang_2018,		
	  title={Density-aware Single Image De-raining using a Multi-stream Dense Network},
	  author={Zhang, He and Patel, Vishal M},
	  booktitle={CVPR},
	  year={2018}
	} 

<p align="center">
<img src="sample_results/121_input.jpg" width="300px" height="200px"/>         <img src="sample_results/121_our.jpg" width="300px" height="200px"/>
<img src="sample_results/38_input.jpg" width="300px" height="200px"/>         <img src="sample_results/38_our.jpg" width="300px" height="200px"/>
</p>



## Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
 
## Installation:
1. Install PyTorch and dependencies from http://pytorch.org (Ubuntu+Python2.7)
   (conda install pytorch torchvision -c pytorch)

2. Install Torch vision from the source.
   (git clone https://github.com/pytorch/vision
   cd vision
   python setup.py install)

3. Install python package: 
   numpy, scipy, PIL, pdb
   
## Demo using pre-trained model
	python test.py --dataroot ./facades/github --valDataroot ./facades/github --netG ./pre_trained/netG_epoch_9.pth   
Pre-trained model can be downloaded at (put it in the folder 'pre_trained'): https://drive.google.com/drive/folders/1VRUkemynOwWH70bX9FXL4KMWa4s_PSg2?usp=sharing

Pre-trained density-aware model can be downloaded at (Put it in the folder 'classification'): https://drive.google.com/drive/folders/1-G86JTvv7o1iTyfB2YZAQTEHDtSlEUKk?usp=sharing

Pre-trained residule-aware model can be downloaded at (Put it in the folder 'residual_heavy'): https://drive.google.com/drive/folders/1bomrCJ66QVnh-WduLuGQhBC-aSWJxPmI?usp=sharing

## Training (Density-aware Deraining network using GT label)
	python derain_train_2018.py  --dataroot ./facades/DID-MDN-training/Rain_Medium/train2018new  --valDataroot ./facades/github --exp ./check --netG ./pre_trained/netG_epoch_9.pth.
	Make sure you download the training sample and put in the right folder

## Density-estimation Training (rain-density classifier)
	python train_rain_class.py  --dataroot ./facades/DID-MDN-training/Rain_Medium/train2018new  --exp ./check_class	

## Testing
	python demo.py --dataroot ./your_dataroot --valDataroot ./your_dataroot --netG ./pre_trained/netG_epoch_9.pth   

## Reproduce

To reproduce the quantitative results shown in the paper, please save both generated and target using python demo.py  into the .png format and then test using offline tool such as the PNSR and SSIM measurement in Python or Matlab.   In addition, please use netG.train() for testing since the batch for training is 1. 
 
## Dataset
Training (heavy, medium, light) and testing (TestA and Test B) data can be downloaded at the following link:
https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view?usp=sharing


## License
Code is under MIT license. 
## Acknowledgments

Great thanks for the insight discussion with [Vishwanath Sindagi](http://www.vishwanathsindagi.com/) and help from [Hang Zhang](http://hangzh.com/)
