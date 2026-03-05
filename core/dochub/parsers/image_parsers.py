import base64
from abc import ABC
from io import BytesIO
from typing import List

import numpy as np
import requests
from PIL import Image
from loguru import logger
from rapid_orientation import RapidOrientation
from rapidocr_onnxruntime import RapidOCR

from config import CONFIG
from dochub.parsers.base import BaseDocumentParser
from dochub.schemas import Chunk, Param, DataType, Document, ChunkType
from dochub.utils.api_utils import gen_image_desc
from dochub.utils.ocr_utils import OCRModelEnum
from utils.i18n import I18NString, Language


image_description_prompt = I18NString({
    Language.ZH: "请详细地描述这张图片，如果图上有文字请告诉我文字是什么",
    Language.EN: "Please describe this image in detail, and if there is any text in the image, please tell me what it says",
})


class BaseImageParser(BaseDocumentParser, ABC):
    target_content_type = "image/"
    target_file_ext = ["png", "jpg", "jpeg"]


class OCRImageParser(BaseImageParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })
    params = [
        Param(name="ocr_model", display_name="OCR模型", required=True, data_type=DataType.STRING,
              default=OCRModelEnum.RapidOCR),
        # 仅使用 ocr 或 vl 一种解析方式。
        Param(name="vl_enable", display_name="开启VL解析", required=True, data_type=DataType.BOOLEAN, default=False),
    ]

    def __init__(self, target: Document, **kwargs):
        super().__init__(target, **kwargs)
        self.ocr_model = self.kwargs.get("ocr_model", OCRModelEnum.RapidOCR)
        self.vl_enable = self.kwargs.get("vl_enable", False)

    def _validate_document(self):
        assert self.target.content_type.startswith(self.target_content_type),\
            f"Invalid content type {self.target.content_type}"

    def _parse_impl(self) -> None:
        self._report_progress(1)

        if self.vl_enable:
            # 开启异步 vl 解析
            self._append_content_chunks(self._vl_parse())
        else:
            self._append_content_chunks(*self._ocr_parse())
        self._report_progress(100)

        cover = self._make_cover(200, 150)
        self._set_attribute_chunk(metadata={"cover": cover})
        self._report_progress(100)

    def _ocr_parse(self) -> List[Chunk]:
        orient_engine = RapidOrientation()
        with Image.open(self.target.physical_path, 'r') as f:
            # 图片方向矫正
            orient, _ = orient_engine(np.array(f))
            if orient == '90':
                rotated_f = f.rotate(270)
            elif orient == '180':
                rotated_f = f.rotate(180)
            elif orient == '270':
                rotated_f = f.rotate(90)
            else:
                rotated_f = f

            if self.ocr_model == OCRModelEnum.ExternalOCR:
                headers = {}
                if api_key := CONFIG["ocr"]["api_key"]:
                    headers["Authorization"] = f"Bearer {api_key}"

                with BytesIO() as output:
                    rotated_f.save(output, format="PNG")
                    png_bytes = output.getvalue()
                files = [("files", ("picture", png_bytes))]
                resp = requests.post(CONFIG["ocr"]["api_url"], files=files, headers=headers).json()
                ocr_result = resp[0]
            else:
                ocr_engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
                ocr_result, _ = ocr_engine(np.array(rotated_f))

            if ocr_result is not None:
                # FIXME 当前将所有文字拼接成一个chunk
                return [Chunk(content=" ".join([text for _, text, _ in ocr_result]), type=ChunkType.TEXT)]
            else:
                return []

    def _vl_parse(self) -> Chunk:
        with open(self.target.physical_path, 'rb') as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_mime = f"image/{self.target.doc_name.split(".")[-1]}"
        image_desc = gen_image_desc(image_base64, image_mime)
        return Chunk(content=image_desc, type = ChunkType.TEXT)


    def _make_cover(self, width, height) -> str:
        """
            创建一个图片的封面，使其适应指定的小框尺寸。

            Args:
                image_path (str): 原始图片的路径。
                output_path (str): 封面图片的保存路径。
                width (int): 目标封面的宽度。
                height (int): 目标封面的高度。
            """
        try:
            # 打开原始图片
            img = Image.open(self.target.physical_path)
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height

            if width / height > aspect_ratio:
                # 小框更宽，以高度为基准调整宽度
                new_width = int(height * aspect_ratio)
                new_height = height
            else:
                # 小框更高，以宽度为基准调整高度
                new_width = width
                new_height = int(width / aspect_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # 创建一个背景为白色的小框
            cover = Image.new("RGB", (width, height), "white")
            # 将调整大小后的图片居中粘贴到小框中
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2
            cover.paste(img, (x_offset, y_offset))

            bytes_buffer = BytesIO()
            cover.save(bytes_buffer, format="JPEG")
            bytes_buffer.seek(0)
            compressed_image = bytes_buffer.read()
            bytes_buffer.close()
            return base64.b64encode(compressed_image).decode("utf-8")
        except Exception as e:
            logger.error(f"发生错误: {e}")

    @staticmethod
    def is_adjacent(box1, box2, threshold):
        """
        Checks if two bounding boxes are adjacent based on a distance threshold.
        """
        a, b = box1[0], box2[0]
        # Calculate the distances between the edges of the boxes
        x_dist = min(abs(a[0][0] - b[3][0]), abs(a[3][0] - b[0][0]))
        y_dist = min(abs(a[0][1] - b[3][1]), abs(a[3][1] - b[0][1]))

        # Check if either distance is below the threshold
        return x_dist <= threshold or y_dist <= threshold

    @staticmethod
    def merge_boxes(box1, box2):
        """
        Merges two bounding boxes into a single box.
        """
        a, b = box1[0], box2[0]
        text = box1[1] + " " + box2[1]
        conf = min(box1[2], box2[2])
        return [
            [min(a[0][0], b[0][0]), min(a[0][1], b[0][1])],  # new left-top
            [max(a[1][0], b[1][0]), min(a[1][1], b[1][1])],  # new right-top
            [max(a[2][0], b[2][0]), max(a[2][1], b[2][1])],  # new right-bottom
            [min(a[3][0], b[3][0]), max(a[3][1], b[3][1])]  # new left-bottom
        ], text, conf
