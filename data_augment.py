# -*- coding:utf-8 -*-
#
# author: dreamer
# date: 20201119

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from urllib.request import urlopen
import pandas as pd
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from albumentations import (
    HorizontalFlip, # pick
    ShiftScaleRotate, # pick
    RandomSunFlare, # pick
    Resize, # pick
    RandomBrightnessContrast, # pick
    GaussianBlur, # pick
    CenterCrop, # pick
    RandomCrop,
    Crop,
    Compose
)



def save_xml(img_id, bbox, labels, save_dir='./Annotations', width=1609, height=500, channel=3):
    '''
        this is for coco style bbox
        将CSV中的一行
        000000001.jpg [[1,2,3,4],...]
        转化成
        000000001.xml
        
        :param image_name:图片名
        :param bbox:对应的bbox
        :param names:   name
        :param save_dir:
        :param width:这个是图片的宽度，博主使用的数据集是固定的大小的，所以设置默认
        :param height:这个是图片的高度，博主使用的数据集是固定的大小的，所以设置默认
        :param channel:这个是图片的通道，博主使用的数据集是固定的大小的，所以设置默认
        :return:
    '''
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_id + '.jpg'#image_name

    # node_filename.text = json_name.replace('json', 'jpg')#image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for (x, y, w, h), name in zip(bbox, labels):
        left, top, right, bottom = x, y, x + w, y + h
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = name
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, img_id + '.xml')    #image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return

# 用于图片上的边界框和类别 labels 的可视化函数
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR,
                lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
#     plt.imshow()

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})


augment_toolkit = [HorizontalFlip(p=1), # pick
    ShiftScaleRotate(), # pick
    RandomSunFlare(src_radius=100, p=1), # pick
    # Resize, # pick
    RandomBrightnessContrast(p=1), # pick
    GaussianBlur(p=1), # pick
    CenterCrop(height=608, width=608, p=1),]# pick]

jpgs_folder = '/home/dreamer/workspace/RongWen/data/Processing/VOCdevkit/VOC2012/decode/add/jpgs'

anno = '/home/dreamer/workspace/RongWen/scripts/rongwen20201117.csv'
anno_csv = pd.read_csv(anno)

aug_jpgs = '/home/dreamer/workspace/RongWen/data/Processing/VOCdevkit/VOC2012/decode/add/aug_jpgs'
aug_annos = '/home/dreamer/workspace/RongWen/data/Processing/VOCdevkit/VOC2012/decode/add/aug_anno'
label_id_to_name = {1: 'traffic cone', 2: 'fence'}

for jpg_file in os.listdir(jpgs_folder):
    image = cv2.imread(os.path.join(jpgs_folder, jpg_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_id = jpg_file.split('.')[0]

    # get bbox info from csv
    bbox = []
    for box in anno_csv[anno_csv['image_id'] == img_id]['bbox']:
        # print(box)
        temp = box[1:-1].split(',')
        temp = [float(i) for i in temp]
        bbox.append(temp)

    # get the label info from csv
    labels = list(anno_csv[anno_csv['image_id'] == img_id]['source'])
    label_dict = {'traffic cone': 1, 'fence': 2}

    labels = [label_dict[label] for label in labels]

    # generating annotations
    label_id_to_name = {1: 'traffic cone', 2: 'fence'}
    annotations = {'image': image, 'bboxes': bbox, 'category_id': labels}

    for i, augment in enumerate(augment_toolkit):
        print(i)
        if i == 5:
            if image.shape[0] < 608  or image.shape[1] < 608:
                continue
        aug = get_aug([augment])
        augmented = aug(**annotations)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['category_id']
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        aug_id = img_id + '_' + str(i)
        cv2.imwrite(os.path.join(aug_jpgs, aug_id + '.jpg'), aug_img)
        aug_labels = [label_id_to_name[index] for index in aug_labels]
        height, width = aug_img.shape[0], aug_img.shape[1]
        save_xml(aug_id, aug_bboxes, aug_labels, save_dir=aug_annos, width=width, height=width, channel=3)
        print(aug_id)