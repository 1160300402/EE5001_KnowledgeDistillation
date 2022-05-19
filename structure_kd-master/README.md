#  Structured Knowledge Distillation

This project is used the code structure from (https://github.com/irfanICMLL/structure_knowledge_distillation), based on paper [Structured Knowledge Distillation for Dense Prediction](https://arxiv.org/pdf/1903.04197.pdf).
It is an extension of paper [Structured Knowledge Distillation for Semantic Segmentation](https://www.zpascal.net/cvpr2019/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) (accepted for publication in [CVPR'19](http://cvpr2019.thecvf.com/), oral).

## Dataset
1.[NYUdv2]Download From (only 13 classes):
    test source: http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
    train source: http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
    test_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
    train_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz

2.[Cityscape dataset](https://www.cityscapes-dataset.com/)

## Results

Student net(Resnet18), on dataset Cityscapes and Nyudv2.
mIou in Nyudv2: before and after distillation is 37.8/38.4
mIou in Cityscapes : before and after distillation is 44.7/45.7
 
## Structure of this repository
This repository is organized as:
* [ckpt](/ckpt/) Logs and student net of dataset Cityscapes
* [libs](/libs/) This directory contains the inplaceABNSync modes.
* [dataset](/dataset/) This directory contains the dataloader for different datasets.
* [network](/network/) This directory contains a model zoo for network models.
* [snapshots](/snapshots/) This directory contains student net of dataset Nyudv2.
* [utils](/utils/) This directory contains api for calculating the distillation loss.


## Pre-trained model and Performance on other tasks
Pretrain models for three tasks can be found here:

| Task |Dataset| Network |Method | Evaluation Metric|Link|
| -- | -- |-- | -- |-- |-- |
| Semantic Segmentation |Cityscapes| ResNet18|Baseline|miou: 69.10 |-|
| Semantic Segmentation |Cityscapes| ResNet18|+ our distillation|miou: 75.3 |[link](https://cloudstor.aarnet.edu.au/plus/s/uL3qO51A4qxY6Eu) |


### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

## Train script
Download the pre-trained [teacher weight](https://cloudstor.aarnet.edu.au/plus/s/tFjYfBJiarVi0pG):

If you want to reproduce the ablation study in our paper, please modify is_pi_use/is_pa_use/is_ho_use in the run_train_eval.sh.
sh run_train_eval.sh

## Test script
If you want to test your method on the cityscape test set, please modify the data-dir and resume-from path to your own, then run the test.sh and submit your results to www.cityscapes-dataset.net/submit/ 
sh test.sh

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Yifan Liu](yifan.liu04@adelaide.edu.au) and [Chunhua Shen](chunhua.shen@adelaide.edu.au).
