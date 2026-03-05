from enum import StrEnum
from io import BytesIO
from typing import List

import numpy as np
import requests
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from config import CONFIG
from dochub.schemas import Chunk, ChunkType


rapid_ocr_engine = RapidOCR()

MIN_X = 20
MIN_Y = 35
MAX_SLOPE = 0.4


class OCRModelEnum(StrEnum):
    RapidOCR = "Rapid OCR"
    ExternalOCR = "External OCR"


class BoxesConnector(object):
    def __init__(self, rects, image_w, max_dist=None, overlap_threshold=None, direct="x"):
        self.direct = direct
        self.rects = np.array(rects)
        self.image_w = image_w
        self.max_dist = max_dist  # x或y轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(image_w)]  # 构建imageW个空列表
        if self.direct == "x":
            for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
                if int(rect[0]) < image_w:
                    self.r_index[int(rect[0])].append(index)
                else:  # 边缘的框旋转后可能坐标越界
                    self.r_index[image_w - 1].append(index)
        else:
            for index, rect in enumerate(rects):
                if int(rect[1]) < image_w:
                    self.r_index[int(rect[1])].append(index)
                else:
                    self.r_index[image_w - 1].append(index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        if min(height1, height2) * 1.0 / max(height1, height2) <= 0.65:
            return 0
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        return Yaxis_overlap

    def calc_overlap_for_Xaxis(self, index1, index2):
        # 计算两个框在x轴方向的重合度(Y轴错位程度)
        width1 = self.rects[index1][2] - self.rects[index1][0]
        width2 = self.rects[index2][2] - self.rects[index2][0]
        x0 = max(self.rects[index1][0], self.rects[index2][0])
        x1 = min(self.rects[index1][2], self.rects[index2][2])

        Yaxis_overlap = max(0, x1 - x0) / min(width1, width2)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        if self.direct == "x":
            start = rect[0] + 1
            end = min(self.image_w - 1, rect[2] + self.max_dist)
        else:
            start = rect[1] + 1
            end = min(self.image_w - 1, rect[3] + self.max_dist)

        for left in range(start, end):
            for idx in self.r_index[left]:
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.direct == "x":
                    if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:
                        return idx
                else:
                    if self.calc_overlap_for_Xaxis(index, idx) > self.overlap_threshold:
                        return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []  # 相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any():  # 优先级是not > and > or
                v = index
                sub_graphs.append([v])
                # 级联多个框(大于等于2个)
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][
                        0]  # np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    sub_graphs[-1].append(v)
        return sub_graphs

    @staticmethod
    def get_rect_points(text_boxes):
        x1 = np.min(text_boxes[:, 0])
        y1 = np.min(text_boxes[:, 1])
        x2 = np.max(text_boxes[:, 2])
        y2 = np.max(text_boxes[:, 3])
        return [x1, y1, x2, y2]

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):

            proposal = self.get_proposal(idx)
            if self.direct == "x":
                if proposal >= 0:
                    self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1
            else:
                if proposal > 0:
                    self.graph[idx][proposal] = 1

        sub_graphs = self.sub_graphs_connected()  # sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  # {0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])  # [[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:
            rect_set = self.rects[list(sub_graph)]  # [[228  78 238 128],[240  78 258 128]].....
            rect_set = BoxesConnector.get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


class UprightBox:
    def __init__(self, coordinate: list, text: str):
        self.coordinate = coordinate
        self.text = text


def ocr_parse_grouped(pic_file: str | bytes, slide_num: int, model: OCRModelEnum) -> List[Chunk]:
    """
    进行图片ocr，并横纵向合并文本框
    """
    result, i_height, i_width = _ocr_parse(pic_file, model)
    if result is None or len(result) == 0:
        return []
    upright_boxes, crooked_boxes = _partition_upright_boxes(result)
    # 合并端正的box中的文字并保存成Chunk
    merged_boxes = _link_boxes(upright_boxes, i_height, i_width)
    chunks = []
    for merged_box in merged_boxes:
        chunk = Chunk(
            content=merged_box.text,
            type=ChunkType.IMAGE,
            # position是文本块的左上角和右下角的坐标，e.g. [144, 5, 192, 25]
            metadata={"positions": merged_box.coordinate, "pages": str(slide_num)}
        )
        chunks.append(chunk)

    # 保存斜的box中的文字,
    for crooked_box in crooked_boxes:
        chunk = Chunk(
            content=crooked_box[1],
            type=ChunkType.IMAGE,
            # position是文本块的4个角的坐标，e.g. [[9.0, 2.0], [321.0, 11.0], [318.0, 102.0], [6.0, 93.0]]
            metadata={"positions": crooked_box[0], "pages": str(slide_num)}
        )
        chunks.append(chunk)
    return chunks


def _link_boxes(boxes: List[UprightBox], image_h: int, image_w: int) -> List[UprightBox]:
    rects = [box.coordinate for box in boxes]
    # 先横向合并
    scaled_dist = int(image_w * 0.022)
    connector_x = BoxesConnector(rects, image_w, max_dist=scaled_dist, overlap_threshold=0.2)
    rects1 = connector_x.connect_boxes()
    res_boxes = [UprightBox(rect.tolist(), "") for rect in rects1]
    for box in boxes:
        for res_box in res_boxes:
            if in_box(box, res_box):
                res_box.text += box.text
                break

    # show_image = np.zeros([image_h, image_w, 3], np.uint8) + 255
    # for rect in rects1:
    #     cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    #
    # cv2.imshow('res', show_image)
    # cv2.waitKey(0)
    # 再纵向合并
    scaled_dist = int(image_h * 0.022)
    connector_y = BoxesConnector(rects1, image_h, max_dist=scaled_dist, overlap_threshold=0.2, direct="y")
    rects2 = connector_y.connect_boxes()
    res_boxes2 = [UprightBox(rect.tolist(), "") for rect in rects2]
    for box in res_boxes:
        for res_box in res_boxes2:
            if in_box(box, res_box):
                res_box.text += box.text + "\n"
                break
    # show_image = np.zeros([image_h, image_w, 3], np.uint8) + 255
    # for rect in rects2:
    #     cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)
    #
    # cv2.imshow('res', show_image)
    # cv2.waitKey(0)
    return res_boxes2


def in_box(box1: UprightBox, box2: UprightBox) -> bool:
    rec1 = box1.coordinate
    rec2 = box2.coordinate
    return rec1[0] >= rec2[0] and rec1[1] >= rec2[1] and rec1[2] <= rec2[2] and rec1[3] <= rec2[3]


def _partition_upright_boxes(boxes: list) -> list:
    """
    筛选出 "端正" 的文本框，它们采用两点表示法，歪斜的还是四点表示法
    """
    res = []
    upright_boxes = []
    crooked_boxes = []
    for box in boxes:
        points = box[0]
        left_bottom_x = points[3][0]
        left_bottom_y = points[3][1]
        right_bottom_x = points[2][0]
        right_bottom_y = points[2][1]
        alpha = abs(left_bottom_y - right_bottom_y) / (abs(left_bottom_x - right_bottom_x) * 1.0)
        if alpha < MAX_SLOPE:
            upright_boxes.append(
                UprightBox([int(left_bottom_x), int(points[0][1]), int(right_bottom_x), int(right_bottom_y)], box[1]))
        else:
            crooked_boxes.append(box)
    res.append(upright_boxes)
    res.append(crooked_boxes)
    return res


def _ocr_parse(pic_file: str | bytes, model: OCRModelEnum):
    """
    根据``pic_file``解析里面的文字, ``pic_file``是图片的二进制方式打开的字节流或图片路径
    """
    if isinstance(pic_file, bytes):
        pic_file = BytesIO(pic_file)
    image = Image.open(pic_file)
    if len(image.getbands()) > 3:
        # 说明图片是透明底，需要加一个黑底背景
        background = Image.new('RGBA', image.size, (0, 0, 0))
        image = Image.alpha_composite(background, image).convert("RGB")
    elif len(image.getbands()) < 3:
        image = image.convert("RGB")

    if model == OCRModelEnum.ExternalOCR:
        with BytesIO() as output:
            image.save(output, format="PNG")
            png_bytes = output.getvalue()
        headers = {}
        if api_key := CONFIG["ocr"]["api_key"]:
            headers["Authorization"] = f"Bearer {api_key}"
        files = [("files", ("picture", png_bytes))]
        resp = requests.post(CONFIG["ocr"]["api_base"], headers=headers, files=files).json()
        return resp[0], image.height, image.width
    else:
        return rapid_ocr_engine(np.array(image))[0], image.height, image.width
