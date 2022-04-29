#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 18:46
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : my_dataset.py
# @software: PyCharm


import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None,
                 txt_name: str = "train.txt"):
        super(VOCDataSet, self).__init__()

        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"

        # 增强容错能力
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")

        print("self.root:", self.root)
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        print(self.img_root)
        print(self.annotations_root)

        # read train.txt or val.txt
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), f"Not found {txt_path} file"

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.xml_list) > 0, \
            f"in{txt_path} file dose not find any information"

        # read class indict
        json_file = "./pascal_voc_classes.json"
        assert os.path.exists(json_file), f"{json_file} file not exist"

        with open(json_file, "r") as f:
            self.class_dict = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        # read xml
        xml_path = self.xml_list[item]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path)
        if img.format != "JPEG":
            raise ValueError(f"Image {img} format not JPEG")

        boxes = []
        lables = []
        iscrowd = []
        assert "object" in data, f"{xml_path} lack of object information"
        print("data", data)

        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有标注信息得可能有w或h为得情况，会导致回归Loss为Nan
            if xmax <= xmin or ymax <= ymin:
                print(f"warning in {xml_path} xml, there are some bbox w/h<=0!")
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            lables.append(self.class_dict[obj["name"]])

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(lables, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"]=boxes
        target["labels"]=labels
        target["image_id"]=image_id
        target["area"] = area
        target["iscrowd"]=iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析为字典形式
        xml: xml tree obtained by parsing xml file content using xml.etree
        return:
            python dict holding xml contents
        """
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表中
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])

        return {xml.tag: result}


import transforms

if __name__ == "__main__":
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    voc_root = r"G:\DL_DATA"
    dataset = VOCDataSet(voc_root=voc_root, transforms=data_transform)
    print(len(dataset))
    print(dataset.__getitem__(0))
    print(dataset.get_height_and_width(0))
