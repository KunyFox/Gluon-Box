import cv2, os
import numpy as np
import mxnet as mx 

from pycocotools.coco import COCO

ids = [51314, 221693, 118209, 80671, 2157, 5992, 10363] 

coco = COCO('/dataset/coco/annotations/instances_val2017.json')

img_root = '/dataset/coco/val2017'

def vis(id):
    name = coco.imgs[id]['file_name']
    img = cv2.imread(os.path.join(img_root, name))
    anns = coco.imgToAnns[id]
    for ann in anns:
        bbox = ann.get('bbox')
        cat_id = ann.get('category_id')
        cat = coco.cats[cat_id]['name']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), \
                      (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255,255,0), 1)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-10)), \
                      (int(bbox[0]+bbox[2]), int(bbox[1])), (255,255,0), -1)
        cv2.putText(img, cat, (int(bbox[0]), int(bbox[1]-1)), \
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.6, color=(255,255,255), thickness=1)
    cv2.imwrite(name, img)

for id in ids:
    vis(id)