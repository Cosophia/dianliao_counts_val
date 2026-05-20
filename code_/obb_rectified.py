#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : YOLOv8_OBB.py
# @Time          : 2024-07-25 17:33:48
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : YOLOv8 Oriented Bounding Box Inference using ONNX
"""

import os
import time


import math
import random

import onnxruntime as ort
from loguru import logger
import onnx
import ast
import numpy as np
import cv2
from math import sqrt
from functools import partialmethod


logger.disable(__name__)


def scale_polygon(points, scale=1.2):
    points = np.array(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    scaled = (points - center) * scale + center
    return scaled.astype(np.int32)


def rotated_iou(rrect1, rrects):
    """
    计算一个旋转矩形与多个旋转矩形的 IoU，使用 OpenCV 的 rotatedRectangleIntersection
    - rrect1: ((cx, cy), (w, h), angle_deg)
    - rrects: iterable of ((cx, cy), (w, h), angle_deg)
    返回: np.ndarray, shape (M,), 每个 IoU 值
    """
    # 计算第一个矩形面积
    (_, _), (w1, h1), _ = rrect1
    area1 = w1 * h1
    ious = []
    for rrect2 in rrects:
        retval, intersect_pts = cv2.rotatedRectangleIntersection(rrect1, rrect2)
        if retval == cv2.INTERSECT_NONE or intersect_pts is None:
            ious.append(0.0)
        else:
            try:
                hull = cv2.convexHull(intersect_pts, returnPoints=True)
                inter_area = cv2.contourArea(hull)
            except Exception:
                inter_area = 0.0
            (_, _), (w2, h2), _ = rrect2
            area2 = w2 * h2
            union = area1 + area2 - inter_area
            ious.append(inter_area / union if union > 0 else 0.0)
    return np.array(ious, dtype=np.float32)


def batch_rotate_nms_grid(rrects, scores, iou_threshold=0.5, cell_size=None):
    """
    网格优化的批量旋转 NMS (CPU 版)，接受 OpenCV 风格的旋转矩形格式
    - rrects: list or np.ndarray of length N, 每个元素 ((cx, cy), (w, h), angle_deg)
    - scores: (N,) ndarray 或可迭代，置信度
    - iou_threshold: float, IoU 抑制阈值
    - cell_size: float 或 None, 网格大小，可选自动计算
    返回: np.ndarray of keep indices (np.int32)
    """
    N = len(rrects)
    if N == 0:
        return np.array([], dtype=np.int32)

    # 提取中心和尺寸，用于网格分配与近似半径计算
    centers = np.array([rect[0] for rect in rrects], dtype=np.float32)  # (N,2)
    wh = np.array([rect[1] for rect in rrects], dtype=np.float32)       # (N,2)
    # 半径取对角线的一半
    radii = np.sqrt(((wh / 2) ** 2).sum(axis=1))

    # 自动计算网格大小
    if cell_size is None:
        cell_size = np.max(radii) * 2 * iou_threshold

    # 网格坐标映射
    grid_coords = (centers / cell_size).astype(int)
    grid_map = {}
    for idx, coord in enumerate(map(tuple, grid_coords)):
        grid_map.setdefault(coord, []).append(idx)

    # 按分数降序
    scores_arr = np.array(scores, dtype=np.float32)
    order = np.argsort(scores_arr)[::-1]
    suppressed = np.zeros(N, dtype=bool)
    keep = []

    # 3×3 邻域偏移
    offsets = np.array([(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1)], dtype=int)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        # 当前矩形和网格
        current_rrect = rrects[idx]
        grid = tuple(grid_coords[idx])
        # 收集邻域候选索引
        neighbor_cells = [ (grid[0]+dx, grid[1]+dy) for dx, dy in offsets ]
        candidates = []
        for cell in neighbor_cells:
            candidates.extend(grid_map.get(cell, []))
        candidates = np.unique(candidates)
        # 排除已抑制与自身
        mask = ~suppressed[candidates]
        mask &= (candidates != idx)
        candidates = candidates[mask]
        if candidates.size == 0:
            continue
        # 精确计算 IoU 并抑制
        candidate_rrects = [rrects[i] for i in candidates]
        ious = rotated_iou(current_rrect, candidate_rrects)
        suppress_idx = candidates[ious > iou_threshold]
        suppressed[suppress_idx] = True

    return np.array(keep, dtype=np.int32)

# 示例：
# from math import pi
# rrects = [((100,100),(40,20),30 * 180/pi), ((110,105),(42,22),28 * 180/pi), ...]
# scores = [0.9, 0.85, ...]
# keep_idx = batch_rotate_nms_grid(rrects, scores, iou_threshold=0.5)

# 示例用法
# boxes = np.array([[100,100,40,20,30], [110,105,42,22,28], ... ])\# scores = np.array([0.9, 0.85, ...])\# keep_idx = batch_rotate_nms_grid(boxes, scores, iou_threshold=0.5)


# 设置每个进程的最大显存占用比例（4GB）
def point_nms_batch(centers, radii, scores, threshold=0.5, cell_size=None):
    """
    高效向量化 Point-NMS (支持网格优化)
    - centers: (N,2) ndarray, 中心坐标
    - radii:   (N,)  ndarray, 半径
    - scores:  (N,)  ndarray, 置信度
    - threshold: float, 抑制条件：距离 < min(r_i, r_j) * threshold
    - cell_size: float, 网格大小（可选，自动计算）
    返回: 保留的索引 (np.int32)
    """
    # 输入检查与转换
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    radii = np.ascontiguousarray(radii, dtype=np.float32)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    N = centers.shape[0]
    if N == 0:
        return np.array([], dtype=np.int32)

    # 1. 自动计算网格大小（若未提供）
    if cell_size is None:
        cell_size = np.max(radii) * 2 * threshold  # 确保最大对象可被网格覆盖

    # 2. 将点分配到网格
    grid_coords = (centers / cell_size).astype(int)
    grid_map = {}
    for idx, (x, y) in enumerate(grid_coords):
        grid_map.setdefault((x, y), []).append(idx)

    # 3. 按分数降序排序
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(N, dtype=bool)
    keep = []

    # 4. 预生成邻域偏移（3x3网格）
    neighbor_offsets = np.array([(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1)], dtype=int)

    # 5. 主循环（向量化优化）
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        current_center = centers[idx]
        current_radius = radii[idx]

        # 5.1 计算当前点所在的网格及邻域
        current_grid = grid_coords[idx]
        neighbor_grids = current_grid + neighbor_offsets

        # 5.2 收集所有邻域内的候选点索引
        candidates = []
        for grid in neighbor_grids:
            candidates.extend(grid_map.get(tuple(grid), []))
        candidates = np.unique(candidates)  # 去重避免重复计算

        # 5.3 过滤已抑制点和自身
        candidates = candidates[~suppressed[candidates] & (candidates != idx)]
        if len(candidates) == 0:
            continue

        # 5.4 向量化计算距离与阈值
        dists = np.linalg.norm(current_center - centers[candidates], axis=1)
        min_radii = np.minimum(current_radius, radii[candidates])
        suppress_mask = dists < (min_radii * threshold)

        # 5.5 执行抑制
        suppressed[candidates[suppress_mask]] = True

    return np.array(keep, dtype=np.int32)

def point_nms(centers, radii, scores, threshold=0.5, cell_size=None):
    N = len(centers)
    if N == 0:
        return np.array([], dtype=np.int32)
    centers = np.array(centers, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    if cell_size is None:
        cell_size = max(radii) * 2 * threshold
    cell_map = {}
    coords = (centers / cell_size).astype(int)
    for i, c in enumerate(coords):
        key = (c[0], c[1])
        cell_map.setdefault(key, []).append(i)
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = np.zeros(N, dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        cx_cell, cy_cell = coords[idx]
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbors.extend(cell_map.get((cx_cell+dx, cy_cell+dy), []))
        center_i = centers[idx]
        radius_i = radii[idx]
        for j in neighbors:
            if suppressed[j] or j == idx:
                continue
            dist = np.linalg.norm(center_i - centers[j])
            if dist < min(radius_i, radii[j]) * threshold:
                suppressed[j] = True
    return np.array(keep, dtype=np.int32)

class RotatedBOX:
    def __init__(self, box, score, class_index):
        self.box = box
        self.score = score
        self.class_index = class_index


class ONNXInfer:
    def __init__(self, onnx_model, class_names=None, device='auto', conf_thres=0.25, nms_thres=0.5) -> None:
        self.onnx_model = onnx_model
        self.class_names = class_names
        self.get_onnx_cls(onnx_model)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.device = self._select_device(device)
        self.radius = 1     #点料圈的大小
        logger.info(f"Loading model on {self.device}...")
        so = ort.SessionOptions()
        so.enable_mem_pattern = False  # 禁用内存模式
        so.enable_mem_reuse = False  # 禁用显存重用
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                # 'arena_extend_strategy': 'kNextPowerOfTwo',
                # 'gpu_mem_limit': 6000 * 1024 * 1024,  # 4096 MB
                'cudnn_conv_algo_search': 'HEURISTIC',
                #'cudnn_conv_algo_search': 'DEFAULT',

                # "EXHAUSTIVE", 最优
                # "HEURISTIC",  经验搜索
                # "DEFAULT"     默认 时间最快 性能最差
                "tunable_op_enable": "1",
                "tunable_op_tuning_enable": "0",
                # 'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]


        self.session_model = ort.InferenceSession(
            self.onnx_model,
            #providers=['CPUExecutionProvider'],
            providers=providers,
            sess_options=self._get_session_options()
        )
        # self.session_model.io_binding()
        # io_binding = self.session_model.io_binding()
        # io_binding.bind_input("images", "cuda", 0, np.float16, (2560, 2560))
        # io_binding.bind_output("output", "cuda")
        # self.session_model.run_with_iobinding(io_binding)
        #
        self.input_shape = self.session_model.get_inputs()[0].shape[2:]

        self.size2nms_thre = {'M': 0.65, 'S': 0.65, 'L': 0.30, 'E': 0.25}

    def set_size_category(self, ch: str) -> None:
        ch = ch.upper()
        assert ch in 'SMLE', f"Supported size category including {tuple(self.size2nms_thre)}"
        self.nms_thre = self.size2nms_thre[ch]

    def _select_device(self, device):
        """
        Select the appropriate device.
        :param device: 'auto', 'cuda', or 'cpu'.
        :return: List of providers.
        """
        if device == 'cuda' or (device == 'auto' and ort.get_device() == 'GPU'):
            # print("cuda")
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _get_session_options(self):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False

        # sess_options.cudnn_conv_algo_search = "HEURISTIC"
        # sess_options.tunable_op_enable = 1
        # # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        #   # sess_options.intra_op_num_threads = 4
        # #sess_options.enable_mem_pattern = True
        # # sess_options.enable_profiling = True
        return sess_options

    # def preprocess(self, img):
    #     """
    #     Preprocess the image for inference.
    #     :param img: Input image.
    #     :return: Preprocessed image blob, original image width, and original image height.
    #     """
    #     logger.info(
    #         "Preprocessing input image to [1, channels, input_w, input_h] format")
    #
    #     height, width = img.shape[:2]
    #
    #     length = max(height, width)
    #     image = np.zeros((length, length, 3), np.uint8)
    #     image[0:height, 0:width] = img
    #     org = image.copy()
    #     #cv2.imwrite("pad_img.png", org)
    #     #针对动态尺寸
    #     if self.session_model.get_inputs()[0].shape[2] == 'height':
    #     # if self.input_shape[0] == 'height':
    #         # self.input_shape = ((length + (32 - length % 32))*4, (length + (32 - length % 32))*4)
    #         self.input_shape = (
    #             (length + 31) // 32 * 32,  # 向上取整到 32 的倍数
    #             (length + 31) // 32 * 32
    #         )
    #         # self.input_shape = ((length + (32 - length % 32)), (length + (32 - length % 32)))
    #         # self.input_shape = (1824, 1824)
    #
    #     # input_shape = self.session_model.get_inputs()[0].shape[2:]
    #     # self.input_shape = (64, 64)
    #     logger.debug(f"Input shape: {self.input_shape}")
    #     image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_LINEAR)
    #
    #     # cv2.imwrite("111.png",image)
    #     image = np.transpose(image, (2, 0, 1)) / 255.0
    #     image = np.expand_dims(image, axis=0)
    #     #blob = image.astype(np.float32) #fp16
    #     blob = image.astype(np.float32) #fp16
    #
    #     # blob = cv2.dnn.blobFromImage(
    #     #     image, scalefactor=1 / 255, size=tuple(self.input_shape), swapRB=True)
    #     # logger.info(f"Preprocessed image blob shape: {blob.shape}")
    #
    #     return blob, org, width, height

    def preprocess(self, img):
        logger.info("Preprocessing input image to [1, channels, input_w, input_h] format")
        self.input_shape = (2560, 2560)
        height, width = img.shape[:2]
        length = max(height, width)
        image = np.ones((length, length, 3), np.uint8) * 255
        image[0:height, 0:width] = img
        padded_image = image.copy()

        if (max(height, width) > self.input_shape[0]):
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]),
                               interpolation=cv2.INTER_LINEAR)
        else:
            image = np.ones((self.input_shape[0], self.input_shape[1], 3), np.uint8)*255
            pad_height, pad_width = padded_image.shape[:2]
            image[0:pad_height, 0:pad_width] = padded_image
            padded_image = image.copy()

        logger.debug(f"Input shape: {self.input_shape}")

        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        # cv2.imwrite("pad2560.png", padded_image)
        return image, padded_image, width, height


    def predict(self, img):
        """
        Perform inference on the image.
        :param img: Input image.
        :return: Inference results.
        """
        blob, resized_image, org_width, org_height = self.preprocess(img)
        #blob = blob.astype(np.float16)
        blob = blob.astype(np.float32)

        # print(resized_image.shape, orig_height, orig_height)
        inputs = {self.session_model.get_inputs()[0].name: blob}
        try:
            infer_start_time = time.time()
            outputs = self.session_model.run(None, inputs)
            infer_end_time = time.time()
            logger.info(f"onnxruntime  {(infer_end_time - infer_start_time)} s")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

        return self.postprocess(outputs, resized_image, org_width, org_height)

    def postprocess(self, outputs, resized_image, orig_width, orig_height):
        """
        Postprocess the model output.
        :param outputs: Model outputs.
        :param resized_image: Resized image used for inference.
        :param orig_width: Original image width.
        :param orig_height: Original image height.
        :return: List of RotatedBOX objects.
        """
        rotated = False
        output_data = outputs[0]
        logger.info(
            f"Postprocessing output data with shape: {output_data.shape}")

        # input_shape = self.session_model.get_inputs()[0].shape[2:]
        x_factor = resized_image.shape[1] / float(self.input_shape[1])
        y_factor = resized_image.shape[0] / float(self.input_shape[0])

        flattened_output = output_data.flatten()
        reshaped_output = np.reshape(
            flattened_output, (output_data.shape[1], output_data.shape[2])).T


        num_classes = len(self.class_names)
        postprocess_start_time = time.time()

        # 提前计算一些常量
        pi_half = 0.5 * math.pi
        pi_three_quarters = 0.75 * math.pi
        pi = math.pi

        # 初始化列表
        nms_rotated_boxes = []
        detected_boxes = []
        rotated_boxes = []
        remain_angle = []
        confidences = []
        centers = []
        radii = []

        # 向量化操作
        class_scores_all = reshaped_output[:, 4:4 + num_classes]
        class_ids = np.argmax(class_scores_all, axis=1)
        confidence_scores = class_scores_all[np.arange(len(class_scores_all)), class_ids]

        # 过滤掉低置信度的检测
        mask = confidence_scores > self.conf_thres
        filtered_detections = reshaped_output[mask]
        filtered_class_ids = class_ids[mask]
        filtered_confidence_scores = confidence_scores[mask]

        # 提前计算缩放因子
        scaling_factors = np.array([x_factor, y_factor, x_factor, y_factor])

        # 遍历过滤后的检测结果
        for detection, class_id, confidence_score in zip(filtered_detections, filtered_class_ids,
                                                         filtered_confidence_scores):
            cx, cy, width, height = detection[:4] * scaling_factors

            # 筛选超出图像区域框
            # if (cx - width / 2 < 0 or cx + width / 2 > orig_width or
            #         cy - height / 2 < 0 or cy + height / 2 > orig_height):
            #     # logger.info(
            #     #     f"Skipping box at center ({cx}, {cy}) with size ({width}, {height}) as it exceeds image boundaries")
            #     continue

            angle = detection[4 + num_classes]

            box = ((cx, cy), (width, height), angle * 180 / math.pi)
            detected_boxes.append(cv2.boundingRect(cv2.boxPoints(box)))

            # 调整角度
            if pi_half <= angle <= pi_three_quarters:
                angle -= pi

            box = ((cx, cy), (width, height), angle * 180 / pi)

            rotated_box = RotatedBOX(box, confidence_score, class_id)
            remain_angle.append(angle)
            rotated_boxes.append(rotated_box)
            confidences.append(confidence_score)
            nms_rotated_boxes.append(box)

            centers.append((cx, cy))
            radii.append(0.5 * math.hypot(width, height))

        # 计算前十个目标框的高宽比，1：5就用rotated
        top_n = 300  # 获取前十个框
        ratios = []  # 用于存储高宽比
        areas = []

        # 从nms_rotated_boxes中提取前十个框并计算高宽比
        for i, box in enumerate(nms_rotated_boxes[:top_n]):
            # box 是一个 tuple: ((cx, cy), (width, height), angle)
            _, (width, height), _ = box
            # 计算高宽比
            if height > width:
                ratio = height / width if width != 0 else 0  # 防止除以零
            else:
                ratio = width / height if height != 0 else 0  # 防止除以零

            areas.append(width * height)
            ratios.append(ratio)

        # 计算平均高宽比
        average_ratio = sum(ratios) / len(ratios) if ratios else 0

        # 计算平均面积
        average_area = sum(areas) / len(areas) if areas else 0

        if (average_area  <  100):
            self.radius = 1
        if (100 < average_area  <  500):
            self.radius = 2
        if (500 < average_area  <  1000):
            self.radius = 5
        if (average_area > 1000):
            self.radius = 8


        # print(f"Average Height/Width Ratio of top {top_n} boxes: {average_ratio}")
        # 检查高宽比是否大于 1:6 或 6:1
        if average_ratio > 3 or average_ratio < 1 / 3:
            rotated = True
            logger.info(f"use rotated nms...")

        org_angle = remain_angle
        # rotated = True

        nms_start_time = time.time()
        if len(detected_boxes) != 0:
        # if len(detected_boxes) == 0:
            # 旋转框nms
            if rotated == True:
                # nms_indices = batch_rotate_nms_grid(nms_rotated_boxes, confidences)
                nms_indices = point_nms_batch(centers, radii,  confidences, self.nms_thres / 2)
                # nms_indices = cv2.dnn.NMSBoxesRotated(
                #     nms_rotated_boxes, confidences, self.conf_thres, self.nms_thres)
            # 无旋转nms
            else:
                nms_indices = point_nms_batch(centers, radii,  confidences, self.nms_thres)
                # nms_indices = cv2.dnn.NMSBoxes(
                #     detected_boxes, confidences, self.conf_thres, self.nms_thres
                # )
                # nms_indices = cv2.dnn.NMSBoxesRotated(
                #     nms_rotated_boxes, confidences, self.conf_thres, self.nms_thres
                # )
            logger.info(f"nms boxes num {len(detected_boxes)}")

            # nms_indices = cv2.dnn.softNMSBoxes(
            #     detected_boxes, confidences, self.conf_thres, self.nms_thres)[1]
            remain_boxes = [rotated_boxes[i] for i in nms_indices.flatten()]
            remain_angle = [remain_angle[i] for i in nms_indices.flatten()]
        else:
            remain_boxes = []
            remain_angle = []
        logger.info(f"Detected {len(remain_boxes)} objects after NMS")
        nms_end_time = time.time()
        logger.info(f"nms time {(nms_end_time - nms_start_time)} s")
        # with open("nms_python.txt", "a") as f:
        #     f.write(f"nms time: {(nms_end_time - nms_start_time)} s \n")
        return (remain_boxes, remain_angle)
        # return (rotated_boxes, org_angle)

    def generate_colors(self, num_classes):
        """
        Generate a list of distinct colors for each class.

        :param num_classes: Number of classes.
        :return: List of RGB color tuples.
        """
        colors = []
        for _ in range(num_classes):
            colors.append((random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)))
        return colors

    def get_onnx_cls(self, modelPath):
        model = onnx.load(modelPath)
        for prop in model.metadata_props:
            if prop.key == "names":
                str_cls = prop.value
        string_data = str_cls.replace('‘', "'").replace('’', "'")
        category_dict = ast.literal_eval(string_data)
        self.class_names = list(category_dict.values())

    def drawshow(self, original_image, detected_boxes):
        """
        Draw detected bounding boxes and labels on the image and display it.

        :param original_image: The input image on which to draw the boxes.
        :param detected_boxes: List of detected RotatedBOX objects.
        :param class_labels: List of class labels.
        """
        # Generate random colors for each class
        detected_boxes, angles = detected_boxes
        num_classes = len(self.class_names)
        colors = self.generate_colors(num_classes)
        # print(len(detected_boxes))
        for idx, detected_box in enumerate(detected_boxes):
            box = detected_box.box
            points = cv2.boxPoints(box)

            class_id = detected_box.class_index
            # print(class_id)

            center = box[0]  # box[0] 是矩形的中心坐标
            #cv2.circle(original_image, (int(center[0]), int(center[1])), self.radius, colors[class_id], -1)
            cv2.circle(original_image, (int(center[0]), int(center[1])), self.radius, (0, 255, 255), -1)

            # # Rescale the points back to the original image dimensions
            points[:, 0] = points[:, 0]
            points[:, 1] = points[:, 1]
            # points = np.int0(points)
            points = points.astype(np.intp)

            class_id = detected_box.class_index


            # Draw the bounding box with the color for the class
            # color = colors[class_id]

            # scaled_points = scale_polygon(points, scale=1.5)
            #
            # #画检测框
            # cv2.polylines(original_image, [scaled_points],
            #               isClosed=True, color=(255, 255, 0), thickness=4)
            # Put the class label text with the same color
            # cv2.putText(original_image, str(f"{angles[idx] *180 / math.pi:1f}"), (points[0][0], points[0][1]),
            #             cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

        # # 在中心位置附近绘制文本
        cv2.putText(original_image, f"{len(detected_boxes)}", (original_image.shape[1] // 2, original_image.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        return original_image


# # # #批量
if __name__ == '__main__':
    files_path = r"C:\Users\Thinkpad\Desktop\cls4\org"
    model_path = r"C:\Users\Thinkpad\Desktop\cls4\cls4.onnx"
    save_dir = r"C:\Users\Thinkpad\Desktop\cls4\res"
    class_names = ['s', 'm', 'l']

    files = os.listdir(files_path)
    img_files = [f for f in files if os.path.splitext(f)[1]  in [".png", ".jpg", ".bmp"]]

    app = ONNXInfer(onnx_model=model_path, device='auto', class_names=class_names, conf_thres=0.25, nms_thres=0.5)

    for img in img_files:
        img_path = os.path.join(files_path, img)
        input = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
        else:
            start_time = time.time()
            predictions = app.predict(input)
            # logger.info(f"Inference results: {predictions}")
            result_img, counts = app.drawshow(input, predictions)

            real_counts = img.split('_')[2]


            if (not os.path.exists(save_dir)):
                os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, img), result_img)
            end_time = time.time()
            logger.info(f"time: {(end_time - start_time)*1000} ms")

# if __name__ == '__main__':
#     # 假设 ONNXInfer 是你自定义的推理类，已正确导入
#     # from your_module import ONNXInfer
#
#     # files_path = r"C:\Users\Thinkpad\Desktop\test-crop"
#     files_path = r"C:\Users\Thinkpad\Desktop\test-crop\test-crop-64x64-crop-area-percent-1\traindata\datasets\images\train"
#     model_path_list = \
#         ["test-crop-64x64-crop-area-percent-1-2",
#          "test-crop-64x64-crop-area-percent-1-1"]
#
#     base_model_path = r"C:\Users\Thinkpad\Desktop\test-crop\model"
#     save_root = r"C:\Users\Thinkpad\Desktop\test-crop\results"
#     class_names = ['s', 'm', 'l']
#
#     # 获取图像文件
#     files = os.listdir(files_path)
#     img_files = [f for f in files if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".bmp"]]
#
#     # 遍历每个模型
#     for model_name in model_path_list:
#         full_model_path = os.path.join(base_model_path, model_name + ".onnx")  # 假设是 .onnx 文件
#         save_dir = os.path.join(save_root, model_name)
#
#         logger.info(f"Loading model: {full_model_path}")
#         app = ONNXInfer(onnx_model=full_model_path, device='gpu', class_names=class_names, conf_thres=0.25,
#                         nms_thres=0.5)
#
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         # 推理每张图像
#         for img in img_files:
#             img_path = os.path.join(files_path, img)
#             input_img = cv2.imread(img_path)
#             if input_img is None:
#                 logger.error(f"Failed to load image: {img_path}")
#                 continue
#
#             start_time = time.time()
#             predictions = app.predict(input_img)
#             result_img = app.drawshow(input_img, predictions, class_names)
#             cv2.imwrite(os.path.join(save_dir, img), result_img)
#             end_time = time.time()
#             logger.info(f"[{model_name}] Processed {img} in {(end_time - start_time) * 1000:.2f} ms")

