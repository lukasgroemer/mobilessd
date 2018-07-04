'''Encode object boxes and labels.'''
import math
import torch

from mobilessd.utils import meshgrid
from mobilessd.utils.box import box_iou, box_nms, change_box_order


class FPNSSDBoxCoder:
    def __init__(self,
                 fm_sizes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
                 anchor_areas=(32 * 32., 64 * 64., 128 * 128., 256 * 256., 341 * 341., 426 * 426., 512 * 512.),
                 aspect_ratios=(1 / 2., 1 / 1., 2 / 1.),
                 scale_ratios=(1., pow(2, 1 / 3.), pow(2, 2 / 3.))):
        '''Compute anchor boxes for each feature map.

        Args:
            fm_sizes: (tuple) feature map sizes of the different fpn layers.
            anchor_areas: (tuple) the anchor areas of the different fpn layers.
            aspect_ratios: (tuple) different aspect ratios of the anchors areas
            scale_ratios: (tuple) different scale_ratios of the anchor areas

        '''
        self.fm_sizes = fm_sizes
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.anchor_boxes = self._get_anchor_boxes(input_size=torch.tensor([512., 512.]))

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          anchor_boxes: (tensor) anchor boxes for each feature map. Each of size [#anchors,4],
            where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.fm_sizes)
        anchor_wh = self._get_anchor_wh()
        fm_sizes = torch.tensor(self.fm_sizes).float()

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)
            wh = anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy - wh / 2., xy + wh / 2.], 3)  # [x,y,x,y]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj,4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            '''Find the max value index(row & col) of a 2D tensor.'''
            v, i = x.max(0)
            j = v.max(0)[1].item()
            return (i[j], j)

        anchor_boxes = self.anchor_boxes
        ious = box_iou(anchor_boxes, boxes)  # [#anchors, #obj]
        index = torch.empty(anchor_boxes.size(0), dtype=torch.long).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] >= 0.5)
        if mask.any():
            index[mask] = ious[mask].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_box_order(anchor_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [#anchors,4].
          cls_preds: (tensor) predicted conf, sized [#anchors,#classes].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        anchor_boxes = change_box_order(self.anchor_boxes, 'xyxy2xywh')
        xy = loc_preds[:, :2] * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_preds[:, 2:].exp() * anchor_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask]
            score = score[mask]
            # print(box.size())
            # print(score.size())

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.empty_like(keep).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores


def test():
    box_coder = FPNSSDBoxCoder()
    print(box_coder.anchor_boxes.size())
    boxes = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]], dtype=torch.float)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loc_targets, cls_targets = box_coder.encode(boxes, labels)
    print(loc_targets.size(), cls_targets.size())

# test()
