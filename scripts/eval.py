import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import pathmagic

from mobilessd.transforms import resize
from mobilessd.datasets import ListDataset
from mobilessd.evaluations.voc_eval import voc_eval
from mobilessd.models.mobilenetfpn import FPNSSD512, MobileNetV2FPN, FPNSSDBoxCoder


DIR_ROOT = os.path.abspath('./')
DATA_PATH = os.path.join(DIR_ROOT, 'mobilessd/datasets')


print('Loading model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = FPNSSD512(21, fpn_net=MobileNetV2FPN, head_channels=256)
net = torch.nn.DataParallel(net)
state = torch.load(
    './scripts/checkpoint/ckpt.pth', map_location=device)
net.load_state_dict(state["net"])
net.eval()

print('Preparing dataset..')
img_size = 512


def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img, boxes, labels


dataset = ListDataset(root=DATA_PATH + '/voc/VOCdevkit/VOC2007/JPEGImages',
                      list_file=DATA_PATH + '/voc/voc07_test.txt',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
box_coder = FPNSSDBoxCoder()

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('mobilessd/datasets/voc/voc07_test_difficult.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(d)


def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(torch.tensor(inputs, device=device))

        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.1)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print(voc_eval(pred_boxes, pred_labels, pred_scores,
                   gt_boxes, gt_labels, gt_difficults,
                   iou_thresh=0.5, use_07_metric=True))


eval(net, dataset)
