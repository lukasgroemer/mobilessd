import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import os
import argparse

import pathmagic

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mobilessd.models.mobilenetfpn import FPNSSD512, MobileNetV2FPN, FPNSSDBoxCoder

parser = argparse.ArgumentParser(description='PyTorch MobileNetV2 FPNSSD Demo')
parser.add_argument('--image', default="000009.jpg", type=str,
                    help='Image to display, must be existant in root folder of data loader')
parser.add_argument('--score_thresh', '-st', default=0.6, type=float,
                    help='Score theshold, necessary in order to accept a prediction')

args = parser.parse_args()

DIR_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(DIR_ROOT, 'mobilessd/datasets')

classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                "train", "tvmonitor"]

print('Loading model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = FPNSSD512(21, fpn_net=MobileNetV2FPN, head_channels=256)
net = torch.nn.DataParallel(net)
state = torch.load(
    './scripts/checkpoint/ckpt.pth', map_location='cpu')
net.load_state_dict(state["net"])
net.eval()

print('Loading image..')
img = Image.open(DATA_PATH + '/voc/VOCdevkit/VOC2007/JPEGImages/' + args.image)
ow = oh = 512
img = img.resize((ow, oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
x = transform(img)
t = time.time()
loc_preds, cls_preds = net(x.unsqueeze(0))
print ("Prediction took: %fs" % (time.time() - t))

print('Decoding..')
box_coder = FPNSSDBoxCoder()
loc_preds = loc_preds.squeeze().cpu()
cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
boxes, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh=args.score_thresh)

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(img)
for box, label in zip(boxes, labels):
    # Create a Rectangle patch
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    ax.annotate(classes_name[label], xy=(box[0] + 3, box[1] + 3), xytext=(box[0] + 3, box[1] + 3),
                bbox={'facecolor': 'red', 'alpha': 0.9, 'pad': 3}, color="white")

plt.show()
