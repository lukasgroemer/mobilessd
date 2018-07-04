# MobileSSD: a PyTorch Single Shot Detector based on the MobileNetV2 architecture

This library is based on the great [TorchCV](https://github.com/kuangliu/torchcv) library and combines it with the [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2) architecture which aims to greatly reduce the model parameters.

## Install

Get the [voc07 train and test set](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). Copy or symlik the VOC datasets to mobilessd/datasets/voc/VOCdevkit (Scipts must be able to find training and test images in mobilessd/datasets/voc/VOCdevkit/VOC2007/JPEGImages).

This done, you can use train.py, test.py and eval.py from the scripts folder.


## Performance

The mAP of the created model currenlty tops for VOC07 dataset at 53%. 

{'ap': array([0.60172077, 0.66596245, 0.38399861, 0.42450882, 0.14971891,
       0.66367717, 0.75209124, 0.54852008, 0.30904903, 0.50163143,
       0.58313164, 0.49539706, 0.70116495, 0.69394266, 0.6345324 ,
       0.25968499, 0.50607345, 0.58362693, 0.67304679, 0.56539054]), 'map': 0.5348434957870685}

Especially for classes "bottles" and "potted plants" the algorithm still has problems. They have in common that bottles and plants often appear in smaller size and grouped on a single image, therefore next step would be to check small anchors in the bottom layers of the fpn.

