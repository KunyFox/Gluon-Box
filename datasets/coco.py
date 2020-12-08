import os 
import numpy as np 

from collections import defaultdict
from .base import BaseDataset
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval



COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class CocoDataset(BaseDataset):
    def __init__(self,
                 ann_file, 
                 img_root,
                 processer,
                 img_root=None,
                 test_mode=False,
                 filter_empty=True,
                 min_size=32):
        super(CocoDataset, self).__init__(ann_file, 
                                          img_root,
                                          test_mode,
                                          filter_empty,
                                          min_size)

        
    def __len__(self):
        return len(self.img_ids)

    def _load_annotations(self):
        coco = COCO(self.ann_file)
        self.img_ids = list(coco.imgs.keys())
        self.img_infos = coco.imgs 
        self.img_anns = coco.imgToAnns
        self.cat_ids = list(coco.cats.keys())
        self.id2cat = {i:coco.cats[i]['name'] for i in self.cat_ids}
        self.id2label = {id: i for i, id in enumerate(self.cat_ids)}
        if self.filter_empty:
            # filter empty and too small image while self.filter_empty=True
            self._filter() 
        self._parse_anns()

    
    def _parse_anns(self):
        img_anns = defaultdict(dict)

        for id in self.img_ids:
            anns = self.img_anns[id]
            labels, bboxes, ignored_bboxes = [], [], []
            for ann in anns:
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_info['width'], x1 + w)
                y2 = min(img_info['height'], y1 + h)
                if x2 < x1 + 1 or y2 < y1 + 1:
                    continue
                if ann['area'] <= 0:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue

                bbox = [x1, y1, x2, y2]
                if ann.get('iscrowd', False):
                    ignored_bboxes.append(bbox)
                else:
                    bboxes.append(bbox)
                    labels.append(self.cat2label[ann['category_id']])

            if len(bboxes) > 0:
                bboxes = np.array(bboxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.array([], dtype=np.float32)

            if len(ignored_bboxes) > 0:
                ignored_bboxes = np.array(ignored_bboxes, dtype=np.float32)
            else:
                ignored_bboxes = np.zeros((0, 4), dtype=np.float32)
            anns = dict(
                bboxes = mx.nd.from_numpy(bboxes),
                labels = mx.nd.from_numpy(labels),
                ignored_bboxes = mx.nd.from_numpy(ignored_bboxes)
            )
            img_anns[id] = anns 
        self.img_anns = img_anns
            
            

    def _filter(self):
        img_ids = []
        img_infos, img_anns = {}, {} 

        for id in self.img_ids:
            info = self.img_infos[id]
            ann = self.img_anns[id]
            if info['height'] < self.min_size or info['width'] < self.min_size or len(ann) < 1:
                continue
            img_ids.append(id)
            img_infos.update({id:info})
            img_anns.update({id:ann})
        self.img_ids = img_ids
        self.img_infos = img_infos
        self.img_anns = img_anns 

    def prepare_train_img(self, idx):
        img_id = self.img_ids[idx] 
        info = self.img_infos[img_id]
        img_ann = self.img_anns[img_id]
        
        img_info = {}
        img_info.update({'bboxes': img_ann['bboxes']})
        img_info.update({'labels': img_ann['labels']})
        img_info.update({'ignored_bboxes': img_ann['ignored_bboxes']})
        img_info.update({
            'info' = {
                file_name=info['file_name'],
                ori_shape=(info['height'], info['width']),
                img_root=self.img_root
            }
        })

        return self.processer(img_info)
