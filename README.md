# Cascade R-CNN: Delving into High Quality Object Detection

by Zhaowei Cai and Nuno Vasconcelos

This repository is written by Zhaowei Cai at UC San Diego.

## Introduction

This repository implements mulitple popular object detection algorithms, including Faster R-CNN, R-FCN, FPN, and our recently proposed Cascade R-CNN, on the MS-COCO and PASCAL VOC datasets. Multiple choices are available for backbone network, including AlexNet, VGG-Net and ResNet. It is written in C++ and powered by [Caffe](https://github.com/BVLC/caffe) deep learning toolbox. 

Cascade R-CNN is a multi-stage extension of the popular two-stage R-CNN object detection framework. The goal is to obtain high quality object detection, which can effectively reject close false positives. It consists of a sequence of detectors trained end-to-end with increasing IoU thresholds, to be sequentially more selective against close false positives. The output of a previous stage detector is forwarded to a later stage detector, and the detection results will be improved stage by stage. This idea can be applied to any detector based on the two-stage R-CNN framework, including Faster R-CNN, R-FCN, FPN, Mask R-CNN, etc, and reliable gains are available independently of baseline strength. A vanilla Cascade R-CNN on FPN detector of ResNet-101 backbone network, without any training or inference bells and whistles, achieved state-of-the-art results on the challenging MS-COCO dataset.

## Citation

If you use our code/model/data, please cite our paper:

    @inproceedings{cai18cascadercnn,
      author = {Zhaowei Cai and Nuno Vasconcelos},
      Title = {Cascade R-CNN: Delving into High Quality Object Detection},
      booktitle = {CVPR},
      Year  = {2018}
    }


## Benchmarking

We benchmark mulitple detector models on the MS-COCO and PASCAL VOC datasets in the below tables.

1. MS-COCO (Train/Test: train2017/val2017, shorter size: 800 for FPN and 600 for the others)

model     | #GPUs | batch size |lr        | max_iter     | train time| AP | AP50 | AP75 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
VGG-RPN-baseline     | 2 | 4    |3e-3| 100k   |  12.5 hr | 23.6 | 43.9 | 23.0 
VGG-RPN-Cascade     | 2 | 4    |3e-3| 100k   |  15.5 hr | 27.0 | 44.2 | 27.7
Res50-RFCN-baseline     | 4 | 1    |3e-3| 280k   |  19 hr | 27.0 | 44.2 | 27.7 
Res50-RFCN-Cascade     | 4 | 1    |3e-3| 280k   |  22.5 hr | 31.1 | 49.8 | 32.8
Res101-RFCN-baseline     | 4 | 1    |3e-3| 280k   |  29 hr | 30.3 | 52.2 | 30.8 
Res101-RFCN-Cascade     | 4 | 1    |3e-3| 280k   |  30.5 hr | 33.3 | 52.0 | 35.2
Res50-FPN-baseline     | 8 | 1    |5e-3| 280k   |  32 hr | 36.5 | 58.6 | 39.2 
Res50-FPN-Cascade     | 8 | 1    |5e-3| 280k   |  36 hr | 40.3 | 59.4 | 43.7
Res101-FPN-baseline     | 8 | 1    |5e-3| 280k   |  37 hr | 38.5 | 60.6 | 41.7 
Res101-FPN-Cascade     | 8 | 1    |5e-3| 280k   |  46 hr | 42.7 | 61.6 | 46.6


2. PASCAL VOC 2007 (Train/Test: 2007+2012trainval/2007test, shorter size: 600)

model     | #GPUs | batch size |lr        | max_iter     | train time| AP | AP50 | AP75 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
Alex-RPN-baseline     | 2 | 4    |1e-3| 45k   |  2.5 hr | 29.4 | 63.2 | 23.7 
Alex-RPN-Cascade     | 2 | 4    |1e-3| 45k   |  3 hr | 38.9 | 66.5 | 40.5
VGG-RPN-baseline     | 2 | 4    |1e-3| 45k   |  6 hr | 42.9 | 76.4 | 44.1 
VGG-RPN-Cascade     | 2 | 4    |1e-3| 45k   |  7.5 hr | 51.2 | 79.1 | 56.3
Res50-RFCN-baseline     | 2 | 2    |2e-3| 90k   |  8 hr | 44.8 | 77.5 | 46.8 
Res50-RFCN-Cascade     | 2 | 2    |2e-3| 90k   |  9 hr | 51.8 | 78.5 | 57.1
Res101-RFCN-baseline     | 2 | 2    |2e-3| 90k   |  10.5 hr | 49.4 | 79.8 | 53.2 
Res101-RFCN-Cascade     | 2 | 2    |2e-3| 90k   |  12 hr | 54.2 | 79.6 | 59.2

**NOTE**. In the above tables, all models have been run at least two times with close results. The training is relatively stable. RPN means Faster R-CNN. The annotations of PASCAL VOC are transformed to COCO format, and COCO API was used for evaluation. The results are different from the official VOC evaluation. If you want to compare the VOC results in publication, please use the official VOC code for evaluation.

## Requirements

1. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.

2. Caffe MATLAB wrapper is required to run the detection/evaluation demo. 

## Installation

1. Clone the CASCADE-RCNN repository, and we'll call the directory that you cloned CASCADE-RCNN into `CASCADE_ROOT`
    ```Shell
    git clone https://github.com/zhaoweicai/cascade-rcnn.git
    ```
  
2. Build CASCADE-RCNN
    ```Shell
    cd $CASCADE_ROOT/
    # Follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j 16

    # If you want to run CASCADE-RCNN detection/evaluation demo, build MATLAB wrapper as well
    make matcaffe
    ```

## Datasets

If you already have a COCO/VOC copy but not as organized as below, you can simply create Symlinks to have the same directory structure. 

### MS-COCO

In all MS-COCO experiments, we use `train2017` for training, and `val2017` (a.k.a. `minival`) for validation, and `test-dev` for final evaluation. Follow [MS-COCO website](http://cocodataset.org/#download) to download images/annotations, and set-up the COCO API.

Assumed that your local COCO dataset copy is at `/your/path/to/coco`, make sure it has the following directory structure:

```
coco
|_ images
  |_ train2017
  |  |_ <im-1-name>.jpg
  |  |_ ...
  |  |_ <im-N-name>.jpg
  |_ val2017
  |_ ...
|_ annotations
   |_ instances_train2017.json
   |_ instances_val2017.json
   |_ ...
|_ MatlabAPI
```

### PASCAL VOC

In all PASCAL VOC experiments, we use VOC2007+VOC2012 `trainval` for training, and VOC2007 `test` for evaluation. Follow [PASCAL VOC website](http://host.robots.ox.ac.uk/pascal/VOC/) to download images/annotations, and set-up the VOCdevkit.

Assumed that your local VOCdevkit copy is at `/your/path/to/VOCdevkit`, make sure it has the following directory structure:

```
VOCdevkit
|_ VOC2007
  |_ JPEGImages
  |  |_ <000001>.jpg
  |  |_ ...
  |  |_ <009963>.jpg
  |_ Annotations
  |  |_ <000001>.xml
  |  |_ ...
  |  |_ <009963>.xml
  |_ ...
|_ VOC2012
  |_ JPEGImages
  |  |_ <2007_000027>.jpg
  |  |_ ...
  |  |_ <2012_004331>.jpg
  |_ Annotations
  |  |_ <2007_000027>.xml
  |  |_ ...
  |  |_ <2012_004331>.xml
  |_ ...
|_ VOCcode
```

## Training CASCADE-RCNN

1. Get the training data
    ```Shell
    cd $CASCADE_ROOT/data/
    sh get_coco_data.sh
    ```
    
    This will download the window files required for the experiments. You can also use the provided MATLAB scripts `coco_window_file.m` under `$CASCADE_ROOT/data/coco/` to generate your own window files.

2. Download the pretrained models on ImageNet. For AlexNet and VGG-Net, the FC layers are pruned and 2048 units per FC layer are remained. In addition, the two FC layers are copied three times for Cascade R-CNN training. For ResNet, the `BatchNorm` layers are merged into `Scale` layers and frozen during training as common practice.
    ```Shell
    cd $CASCADE_ROOT/models/
    sh fetch_vggnet.sh
    ```

3. Multiple shell scripts are provided to train CASCADE-RCNN on different baseline detectors as described in our paper. Under each model folder, you need to change the `root_folder` of the data layer in `train.prototxt` and `test.prototxt` to your COCO path. After that, you can start to train your own CASCADE-RCNN models. Take `vgg-12s-600-rpn-cascade` for example. 
    ```Shell
    cd $CASCADE_ROOT/examples/coco/vgg-12s-600-rpn-cascade/
    sh train_detection.sh
    ```
   Log file will be generated along the training procedure. The total training time depends on the complexity of models and datasets. If you want to quickly check if the training works well, try the light AlexNet model on VOC dataset. 
 
**NOTE**. Occasionally, the training of the Res101-FPN-Cascade will be out of memory. Just resume the training from the latest solverstate.

## Pretrained Models

We only provide the Res50-FPN baseline, Res50-FPN-Cascade and Res101-FPN-Cascade models, for COCO dataset, and Res101-RFCN-Cascade for VOC dataset.

Download pre-trained models
```Shell
cd $CASCADE_ROOT/examples/coco/
sh fetch_cascadercnn_models.sh
``` 
The pretrained models produce exactly the same results as described in our paper.

## Testing/Evaluation Demo

Once the models pretrained or trained by yourself are available, you can use the MATLAB script `run_cascadercnn_coco.m` to obtain the detection and evaluation results. Set the right dataset path and choose the model of your interest to test in the demo script. The default setting is to test the pretrained model. The final detection results will be saved under $CASCADE_ROOT/examples/coco/detections/ and the evaluation results will be saved under the model folder.

You also can run the shell `test_coco_detection.sh` under each model folder for evalution, but it is not identical to the official evaluation. For publication, use the MATLAB script.

## Disclaimer

1. When we were re-implementing the FPN framework and `roi_align` layer, we only referred to their published papers. Thus, our implementation details could be different from the official [Detectron](https://github.com/facebookresearch/Detectron).

If you encounter any issue when using our code or model, please let me know.
