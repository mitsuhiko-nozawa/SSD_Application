import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt
from pathlib import Path
import xml.etree.ElementTree as ET

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2

from .transforms import *

class Anno_xml2list(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        ret = []
        xml = ET.parse(xml_path).getroot()
        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in (pts):
                cur_pixel = int(bbox.find(pt).text) - 1
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets



class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # intをfloat32に変換
                ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                PhotometricDistort(),  # 画像の色調などをランダムに変化
                Expand(color_mean),  # 画像のキャンバスを広げる
                RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
                RandomVerticalFlip(p=0.5), # 追加で実装した物
                #RandomHorizontalFlip(p=0.5), # 追加で実装した物
                ToPercentCoords(),  # アノテーションデータを0-1に規格化
                Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
                SubtractMeans(color_mean)  # BGRの色の平均値を引き算
            ]),
            'val': Compose([
                ConvertFromInts(),  # intをfloatに変換
                Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
                SubtractMeans(color_mean)  # BGRの色の平均値を引き算
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


class TrainDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train もしくは valを指定
        self.transform = transform  # 画像の変形
        self.transform_anno = transform_anno  # アノテーションデータをxmlからリストへ

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height) #np.array([[xmin, ymin, xmax, ymax, label], [...]]

        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, gt, height, width

class TestDataset(data.Dataset):
    def __init__(self, img_list, phase, transform):
        self.img_list = img_list
        self.phase = phase  # train もしくは valを指定
        self.transform = transform  # 画像の変形

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = self.pull_item(index)
        return im

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        anno_list = np.array([0, 1, 2, 3, 4]).reshape(1, 5) # ダミーデータ

        img, boxes, labels = self.transform(img, self.phase, [], anno_list[:, 4])
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        return img

