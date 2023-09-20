# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from fsdet.data import MetadataCatalog
from fsdet.utils import comm
from fsdet.utils.logger import create_small_table

from .evaluator import DatasetEvaluator


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        try:
            self._base_classes = meta.base_classes
            self._novel_classes = meta.novel_classes
        except AttributeError:
            self._base_classes = meta.thing_classes
            self._novel_classes = None
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            rec_novel = defaultdict(list)
            prec_novel = defaultdict(list)
            # TPandFN_novel = defaultdict(list)
            exist_base, exist_novel = False, False
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap= voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

                    if self._base_classes is not None and cls_name in self._base_classes:
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if self._novel_classes is not None and cls_name in self._novel_classes:
                        aps_novel[thresh].append(ap * 100)
                        if thresh == 50:
                            rec_novel[cls_name].append(rec)
                            prec_novel[cls_name].append(prec)
                        exist_novel = True
        
        # print("=================================================================")
        # print("TPs_novel2023年8月24日 17:01:1222222222222")

        # print(TPandFN_novel[cls_name])

        F1_class = defaultdict(list)
        F1_ALL = 0.0
        eps_i = 1e-8
        for Tclass in self._novel_classes:
            # print("=================================================================")
            # print(Tclass)
            # print(rec_novel[Tclass])
            # rec_novel_class = np.sum(rec_novel[Tclass],axis=1)
            # rec_novel_class = np.sum(rec_novel[Tclass],axis=0)
            rec_novel_class = rec_novel[Tclass][0]
            # rec_novel_class = rec_novel[Tclass]
            # print(rec_novel_class)
            # prec_novel_class = np.sum(prec_novel[Tclass],axis=1)
            # prec_novel_class = np.sum(prec_novel[Tclass],axis=0)
            prec_novel_class = prec_novel[Tclass][0]
            
            # prec_novel_class = prec_novel[Tclass]
            # print(prec_novel_class)
            # print("=================================================================")
            # TPandFN_num = 
            # print(TPandFN_novel[Tclass])
            # TPandFN_num = np.sum(TPandFN_novel[Tclass],axis=1)
            # TPandFN_list = TPandFN_novel[Tclass]
            # print("TPandFN_list=================================================================")
            # print(TPandFN_list)


            F1_Score =  2 * ((prec_novel_class * rec_novel_class)/(prec_novel_class + rec_novel_class+eps_i))
            F1_class[Tclass].append(F1_Score) 

        result_str = "F1_score："
        M_Sign = '@'
        result_str = result_str + M_Sign
        result_str_class = ""
        result_str_ALL =""

        for class_i,F1_score_i in F1_class.items():
            for F1_score_i_tmp in F1_score_i :
                print(str(class_i)+"+"+str(F1_score_i_tmp)+'|')
                P_str = str(class_i)+"+"+str(F1_score_i_tmp)+'|'
                result_str = result_str + P_str
                result_str_class = result_str_class + P_str
                F1_ALL = F1_ALL + F1_score_i_tmp
        F1_ALL = F1_ALL/len(self._novel_classes)
        print("Novle+"+str(F1_ALL)) 
        P_str = "Novle+"+str(F1_ALL)     
        result_str = result_str + P_str
        result_str_ALL = result_str_ALL + P_str

        result_str = result_str + M_Sign
        print("=================================================================")

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())), "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75]}
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())), "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75]
            })

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}

        self._logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n"+create_small_table(ret["bbox"]))
        self._logger.info("Novle_F1:\n"+result_str)
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = 0
        obj_struct["difficult"] = 0
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

# 计算AP，即计算面积
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections 【需要计算的类别的txt文件路径】
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations file. 【label的xml文件所在的路径】
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line. 【测试txt文件，里面是每个测试图片的地址，每行一个地址】
    classname: Category name (duh)【需要计算的类别】
    [ovthresh]: Overlap threshold (default = 0.5) 【IOU重叠度 (default = 0.5)】
    [use_07_metric]: Whether to use VOC07's 11 point AP computation 【是否使用VOC07的11点AP计算(default False)】
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt 加载ground truth
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    # 得到的是每一行单个的测试图片的名称
    imagenames = [x.strip() for x in lines]

    # load annots
    # 通过parse_rec对每一个图片进行解析，并将结果保存到object字典当中，在这里是recs
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {} # 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname] #获取当前类别的obj
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)  #  different基本都为0/False.
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)  #初始化一个含有该长度值个False和None的列表，作为标记
        npos = npos + sum(~difficult)   #计算非困难的总数，剔除different的物体，计算总数
        class_recs[imagename] = {"bbox": bbox,
                                 "difficult": difficult,    #大多数都为0
                                 "det": det}

    # read dets
    # read dets 读取某类别预测输出
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    # 图片ID
    image_ids = [x[0] for x in splitlines]
    # IOU值
    confidence = np.array([float(x[1]) for x in splitlines])
    # bounding box数值
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    # 重排bbox，由大概率到小概率。
    BB = BB[sorted_ind, :]
    # 图片重排，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        '''
            1. 如果预测输出的是(x_min, y_min, x_max, y_max)，那么不需要下面的top,left,bottom, right转换
            2. 如果预测输出的是(x_center, y_center, h, w),那么就需要转换
            3. 计算只能使用[left, top, right, bottom],对应lable的[x_min, y_min, x_max, y_max]
        '''
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            # 最大重叠
            ovmax = np.max(overlaps)
            # 最大重合率对应的gt
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            # 计算tp 和 fp个数
            if not R["difficult"][jmax]:
                # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    # 标记为已检测
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0


    # compute precision recall
    fp = np.cumsum(fp)

    tp = np.cumsum(tp)

    # print("fp ================")
    # print(fp)
    # print("tp ================")
    # print(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # print("rec ================")
    # print(rec)
    # print("prec ================")
    # print(prec)

    if len(fp):  # 存在值即为真
        r = rec[-1]
        p = prec[-1]
    else:  # list_temp是空的
        r = 0.0
        p = 0.0
    return r, p,ap