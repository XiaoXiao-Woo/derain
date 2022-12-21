# Derain Methods
## A Decoder-free Transformer-like Architecture for High-efficiency Single Image Deraining (IJCAI Long Oral, 3.7%)
[Xiao Wu](https://scholar.google.com/citations?user=-aFhoQgAAAAJ&hl=zh-CN), [Ting-Zhu Huang](https://scholar.google.com/citations?user=H7El-ZkAAAAJ&hl=zh-CN), [Liang-Jian Deng](https://scholar.google.com/citations?user=TZs9NxkAAAAJ&hl=zh-CN), [Tian-Jing Zhang](https://tianjingzhang.github.io/)

[Paper](https://www.ijcai.org/proceedings/2022/0205.pdf) |
[Video](https://www.ijcai.org/proceedings/2022/video/205)

> **Abstract:** *Despite the success of vision Transformers for the image deraining task, they are limited by computation-heavy and slow runtime. In this work, we investigate Transformer decoder is not necessary and has huge computational costs. Therefore, we revisit the standard vision Transformer as well as its successful variants and propose a novel Decoder-Free Transformer-Like (DFTL) architecture for fast and accurate single image deraining. Specifically, we adopt a cheap linear projection to represent visual information with lower computational costs than previous linear projections. Then we replace standard Transformer decoder block with designed Progressive Patch Merging (PPM), which attains comparable performance and efficiency. DFTL could significantly alleviate the computation and GPU memory requirements through proposed modules. Extensive experiments demonstrate the superiority of DFTL compared with competitive Transformer architectures, e.g., ViT, DETR, IPT, Uformer, and Restormer.* 
<hr />

## Toy Example
![image](https://user-images.githubusercontent.com/15083102/208884898-6368dee0-3fb5-4236-a86c-80fb3623998a.png)

## Training and Evaluation

Training and Testing for Deraining:

<table>
  <tr>
    <th align="left">Derain</th>
    <th align="center">Dataset</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">Rain200L</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
    <td align="center"><a href="">Download</a></td>
  </tr>
  <tr>
    <td align="left">Rain200H</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Link</a></td>
    <td align="center"><a href="">Download</a></td>
  </tr>
  <tr>
    <td>DID</td>
    <td align="center"><a href="https://github.com/hezhangsprinter/DID-MDN">Link</a></td>
    <td align="center"><a href="">Download</a></td>
  </tr>
  <tr>
    <td>DDN</td>
    <td align="center"><a href="https://xueyangfu.github.io/projects/cvpr2017.html">Link</a></td>
    <td align="center"><a href="">Download</a></td>
  </tr>
</table>

**Step1.**
* Download datasets and put it with the following format. 

* Verify the dataset path in `configs/configs.py`.
```
|-$ROOT/data
├── Rain200H
│   ├── train_c
│   │   ├── norain-1.png
│   │   ├── ...
│   ├── test_c
│   │   │   ├── norain-1.png
│   │   │   ├── ...
```

**Step2.** Open codes in your ide,  run the following code:

> python run_derain.py

* A training example：

	run_derain.py
  
	where arch='Restormer', and configs/option_Restormer.py has: 
  
	__cfg.eval__ = False, 
  
	__cfg.workflow__ = [('train', 50), ('val', 1)], __cfg.dataset__ = {'train': 'Rain200H', 'val': 'Rain200H'}
	
* A test example:

	> run_derain_test.py
  
	__cfg.eval__ = True or __cfg.workflow__ = [('val', 1)]

**Run our DFTL**

Currently, you can run run_DFTLW.py or run_DFTLX.py in [Link](https://github.com/XiaoXiao-Woo/derain/tree/main/models/compared_trans/DFTL).

Note: Our project is based on MMCV, but you needn't to install it. More importantly, it can be more easy to introduce more methods.

## Benchmark 
We provide simple pipelines to train/test/inference models for a quick start.

<details open>
<summary>Derain model zoo:
</summary>

* DSC (ICCV'2015)
* GMM (CVPR'2016)
* Clear (TIP'2017)
* DDN (CVPR'2017)
* RESCAN (ECCV'2018)
* NLEDN (ACMMM'2018)
* PReNet (CVPR'2019)
* FBL (AAAI'2020)
* RCDNet (CVPR'2020)
* DualGCN (AAAI'2021)
* IPT (CVPR'2021)
* Uformer (CVPR'2022)
* Restormer (CVPR'2022)

</details>


## Citation
If it is helpful for you, please kindly cite our paper:
```
  @inproceedings{DFTL,
    title     = {A Decoder-free Transformer-like Architecture for High-efficiency Single Image Deraining},
    author    = {Wu, Xiao and Huang, Ting-Zhu and Deng, Liang-Jian and Zhang, Tian-Jing},
    booktitle = {Proceedings of the Thirty-First International Joint Conference on
                 Artificial Intelligence (IJCAI-22)},
    pages     = {1474--1480},
    year      = {2022},
    month     = {7},
    doi       = {10.24963/ijcai.2022/205},
  }
  ```
  



## Contact
Should you have any question, please contact wxwsx1997@gmail.com;


**Acknowledgment:** This code is based on the [MMCV](https://github.com/open-mmlab/mmcv) toolbox and [Restormer](https://github.com/swz30/Restormer). 

## Our Related Works
- "PanCollection" for Remote Sensing Pansharpening, 中国图象图形学报 2022. [Paper](https://liangjiandeng.github.io/papers/2022/deng-jig2022.pdf) | [Code](https://github.com/XiaoXiao-Woo/PanCollection)
- Dynamic Cross Feature Fusion for Remote Sensing Pansharpening, ICCV 2021. [Paper](https://liangjiandeng.github.io/papers/2021/dfcnet2021.pdf) | [Code](https://github.com/XiaoXiao-Woo/UDL)

## License & Copyright
This project is open sourced under GNU General Public License v3.0.
