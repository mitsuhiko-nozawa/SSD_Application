# パッケージのimport
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt

# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

from pathlib import Path
from .transforms import *

# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する
def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    rootpath = Path(rootpath)

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    voc2012_img_dir = Path(rootpath) / "JPEGImages"
    voc2012_anno_dir = Path(rootpath) / "Annotations"
    if not voc2012_img_dir.exists():
        raise Errors.FileNotFound(voc2012_img_dir)
    if not voc2012_anno_dir.exists():
        raise Errors.FileNotFound(voc2012_img_dir)

    # 訓練と検証、それぞれのファイルのID(ファイル名)を取得する
    id_names_root = rootpath / "ImageSets" / "Main"
    train_id_names = id_names_root / "train.txt"
    val_id_names = id_names_root / "val.txt"

    # 訓練データと画像ファイルとアノテーションファイルへのパスリストを作成
    def glob_file_names_from_(id_names_file):
        img_list = []
        anno_list = []
        for line in open(id_names_file):
            file_id = line.strip() # 空白スペースと改行を除去
            img_list.append(str(voc2012_img_dir / f"{file_id}.jpg"))
            anno_list.append(str(voc2012_anno_dir / f"{file_id}.xml"))
        return img_list, anno_list
    ret = glob_file_names_from_(train_id_names)
    train_img_list = ret[0]
    train_anno_list = ret[1]
    ret = glob_file_names_from_(val_id_names)
    val_img_list = ret[0]
    val_anno_list = ret[1]

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# 「XML形式のアノテーション」を、リスト形式に変換するクラス
class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

        Parameters
        ----------
        xml_path : str
            xmlファイルへのパス。
        width : int
            対象画像の幅。
        height : int
            対象画像の高さ。

        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数分のだけ要素を持つ。
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納します
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOCは原点が(1,1)なので1を引き算して（0, 0）に
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # resに[xmin, ymin, xmax, ymax, label_ind]を足す
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# 入力画像の前処理をするクラス
class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
    画像のサイズを512x512にする。
    学習時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (B, G, R)
        各色チャネルの平均値。
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # intをfloat32に変換
                ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                PhotometricDistort(),  # 画像の色調などをランダムに変化
                Expand(color_mean),  # 画像のキャンバスを広げる
                RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
                ###
                RandomVerticalFlip(p=0.5), # 追加で実装した物
                #RandomHorizontalFlip(p=0.5), # 追加で実装した物
                ###
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
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
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


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets