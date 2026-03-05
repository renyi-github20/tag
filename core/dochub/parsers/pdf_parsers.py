import base64
import json
import re
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Optional, List
from zipfile import ZipFile

import camelot
import fitz
import requests
from loguru import logger
from rapid_orientation import RapidOrientation
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

from config import CONFIG
from dochub.parsers.base import BaseDocumentParser
from dochub.schemas import Param, Chunk, DataType, ChunkType
from dochub.utils.api_utils import gen_image_desc
from dochub.utils.text_splitters import RecursiveCharacterTextSplitter
from utils.i18n import I18NString, Language
from utils.token import count_tokens

garbled_char_regex = re.compile(r'[\x00-\x05#%$!&*]')


def clean_text(text: str):
    # 去除latexit的乱码
    left = "<latexit"
    right = "</latexit>"
    start = text.find(left)
    while start != -1:
        end = text.find(right)
        text = text[:start] + text[end + len(right):]
        start = text.find(left)
    return text.strip()


def is_garbled(page_text, garbled_threshold=0.01):
    garbled_char_cnt = len(garbled_char_regex.findall(page_text))
    return garbled_char_cnt > len(page_text) * garbled_threshold or garbled_char_cnt > 5


def merge_bounding_boxes(a, b):
    """
    Merge overlapping or adjacent bounding boxes represented by four vertices.

    Parameters:
    - bboxes: List of lists representing bounding boxes as four vertices
                    (left-top, right-top, right-bottom, left-bottom).

    Returns:
    - List of merged bounding boxes represented by four vertices.
    """
    # Merge the boxes by updating the vertices
    return [
        [min(a[0][0], b[0][0]), min(a[0][1], b[0][1])],  # new left-top
        [max(a[1][0], b[1][0]), min(a[1][1], b[1][1])],  # new right-top
        [max(a[2][0], b[2][0]), max(a[2][1], b[2][1])],  # new right-bottom
        [min(a[3][0], b[3][0]), max(a[3][1], b[3][1])]  # new left-bottom
    ]


def get_bounding_box_area(bbox):
    # Calculate the area of the bounding box
    return (bbox[1][0] - bbox[0][0]) * (bbox[2][1] - bbox[1][1])


class BasePDFParser(BaseDocumentParser, ABC):
    target_content_type = "application/pdf"
    target_file_ext = "pdf"


class GeneralPDFParser(BasePDFParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })
    params = [
        Param(name="chunk_size", display_name="文本块大小", required=True, data_type=DataType.NUMBER, default=500),
        Param(name="chunk_overlap", display_name="文本块间重叠字符数", required=False, data_type=DataType.NUMBER,
              default=100),
    ]

    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.kwargs.get("separators", ["\n\n", "\n", "。", ""]),
            chunk_size=self.kwargs.get("chunk_size", 500),
            chunk_overlap=self.kwargs.get("chunk_overlap", 100)
        )
        self.block_size = self.kwargs.get("block_size", 20)

    def _parse_impl(self) -> None:
        with fitz.open(self.target.physical_path) as doc:
            page_blocks = []
            page_block = []
            for i, page in enumerate(doc):
                if i % self.block_size == 0 and i != 0:
                    page_blocks.append(page_block)
                    page_block = []
                page_block.append(page)
            if page_block:
                page_blocks.append(page_block)

            futures = []
            with ThreadPoolExecutor(max_workers=4) as parse_pool:
                for page_block in page_blocks:
                    futures.append(parse_pool.submit(self._do_parse, page_block))
                for future in self._progress_wrapper(futures, len(futures)):
                    res = future.result()
                    if res:
                        self._append_content_chunks(*res)

    def _do_parse(self, pages) -> List[Chunk]:
        chunks = []
        for page in pages:
            # 通过判断页面内图片的大小，判断当前页是否是扫描图片，进而进行OCR识别
            flag_ocr = False
            for image in page.get_images():
                width, height = image[2], image[3]
                page_width, page_height = page.mediabox_size
                if width >= page_width * 0.8 and height >= page_height * 0.8:
                    chunks += RapidOCRPDFParser.ocr_page(page)
                    flag_ocr = True
                    break
            if flag_ocr:
                continue

            # 普通页面进行文本抽取, 去除图片解析出的乱码
            page_content = clean_text(page.get_text().strip())
            # 如果该页解析出来全是乱码文字，则调用ocr识别
            if is_garbled(page_content):
                chunks += RapidOCRPDFParser.ocr_page(page)
                continue

            if page_content:
                for snippet in self.text_splitter.split_text(page_content):
                    chunk = Chunk(content=snippet, type=ChunkType.TEXT, metadata={"pages": str(page.number + 1)})
                    chunks.append(chunk)

            # 将图片抽取出来，在content中存储base64
            seen_images = set()
            for image_info in page.get_images():
                xref = image_info[0]
                image_obj = page.parent.extract_image(xref)
                # 过滤掉小图
                if image_obj['width'] < 128 and image_obj['height'] < 128:
                    continue
                image_binary = image_obj["image"]
                base64_str = base64.b64encode(image_binary).decode("utf-8")
                # 图片去重
                if base64_str not in seen_images:
                    img_mime = f"image/{image_obj['ext']}"
                    img_desc = gen_image_desc(base64_str, img_mime)
                    chunk = Chunk(content=img_desc, type=ChunkType.IMAGE, metadata={
                        "pages": str(page.number + 1),
                        "image_base64": base64_str,
                        "image_mime": img_mime,
                    })
                    chunks.append(chunk)
                    seen_images.add(base64_str)

        # 识别表格
        try:
            tables = camelot.read_pdf(
                self.target.physical_path,
                pages=",".join([str(p.number + 1) for p in pages]),
                strip_text='\r\n',
                copy_text='vh',
            )
            for table in tables:
                table_repr = table.df.to_csv(sep='|', index=False, header=False)
                chunk = Chunk(content=table_repr, type=ChunkType.TABLE, metadata={"pages": str(table.page)})
                chunks.append(chunk)
        except:
            logger.exception("Failed to extract tables.")
        return chunks


class RapidOCRPDFParser(BasePDFParser):
    name = I18NString({
        Language.ZH: "Rapid OCR",
        Language.EN: "Rapid OCR",
    })

    ocr_engine = RapidOCR()
    orient_engine = RapidOrientation()


    @classmethod
    def ocr_page(cls, page) -> List[Chunk]:
        pix = page.get_pixmap()

        # 文档方向矫正
        orient, _ = cls.orient_engine(pix.tobytes())
        if orient == '90':
            page.set_rotation(270)
        elif orient == '180':
            page.set_rotation(180)
        elif orient == '270':
            page.set_rotation(90)
        rotated_pix = page.get_pixmap()

        ocr_result, _ = cls.ocr_engine(rotated_pix.tobytes())
        if ocr_result is None:
            return []

        # 将 ocr_results 转换成 chunks
        chunks = []
        cur_page = page.number + 1
        cur_content = ""
        cur_bbox = None
        for item in ocr_result:
            bbox, text, _ = item
            cur_content += text + "\n"
            if cur_bbox is None:
                cur_bbox = bbox
            else:
                cur_bbox = merge_bounding_boxes(cur_bbox, bbox)
            # 语义合并
            if text.endswith("。"):
                chunk = Chunk(content=cur_content, type=ChunkType.TEXT,
                              metadata={"pages": str(cur_page), "bbox": cur_bbox})
                chunks.append(chunk)
                cur_content = ""
                cur_bbox = None

        if cur_content != "":
            chunk = Chunk(content=cur_content, type=ChunkType.TEXT,
                          metadata={"pages": str(cur_page), "bbox": cur_bbox})
            chunks.append(chunk)
        return chunks

    def _parse_impl(self) -> None:
        with fitz.open(self.target.physical_path) as doc:
            for page in tqdm(doc, total=len(doc), desc=f"OCR {self.target.doc_name} ({self.target.doc_id})"):
                self._append_content_chunks(*self.ocr_page(page))


class NextGenPDFParser(BasePDFParser):
    name = I18NString({
        Language.ZH: "NextGen",
        Language.EN: "NextGen",
    })
    params = [
        Param(name="chunk_size", display_name="文本块大小", required=True, data_type=DataType.NUMBER, default=500),
    ]

    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.chunk_size = kwargs.get("chunk_size", 500)

    def _retry_nextgen_and_check(self) -> Optional[BytesIO]:
        total_cnt, total_cnt_copy = 2, 2  # 最多尝试两次
        while total_cnt > 0:
            total_cnt -= 1
            logger.info(f"try nextgen-pdf-parser time: {total_cnt_copy - total_cnt}")
            with open(self.target.physical_path, 'rb') as f:
                headers = {}
                if api_key := CONFIG["nextgen_pdf_parser"]["api_key"]:
                    headers["Authorization"] = f"Bearer {api_key}"
                resp = requests.post(CONFIG["nextgen_pdf_parser"]["api_url"],
                                     files={"file": f},
                                     headers=headers,
                                     allow_redirects=False)

                if resp.status_code == 307:
                    resp = requests.post(resp.headers['Location'])
                zip_bytes = BytesIO(resp.content)

                # 获取uid
                content_disposition = resp.headers.get('Content-Disposition')
                uid = ""
                if content_disposition:
                    name = None
                    for part in content_disposition.split(';'):
                        part = part.strip()
                        if part.startswith('filename='):
                            name = part.split('=')[1].strip('"')  # 去除引号
                            break
                    uid = name.split('.')[0] if name else uid

                # 验证zip包合法性
                valid = False
                with ZipFile(zip_bytes, 'r') as myzip:
                    for filename in myzip.namelist():
                        if filename.endswith("content_list.json"):
                            valid = True
                            break
                if valid:
                    zip_bytes.seek(0)
                    return zip_bytes
        return None

    def _parse_impl(self) -> None:
        zip_bytes = self._retry_nextgen_and_check()
        if zip_bytes is None:
            raise RuntimeError("error happens in nextgen-pdf-parser")
        content_list = None
        with ZipFile(zip_bytes, 'r') as myzip:
            for filename in myzip.namelist():
                if filename.endswith("content_list.json"):
                    with myzip.open(filename, "r") as f:
                        content_list = json.loads(f.read())
                    break

            text_buffer = ""
            last_page = 0
            for idx, obj in self._progress_wrapper(enumerate(content_list), len(content_list)):
                if "type" not in obj:
                    continue
                item = obj.copy()
                page_idx = item.pop("page_idx", last_page)
                item["pages"] = str(int(page_idx) + 1)

                try:
                    if item["type"] == "text":
                        text = item.pop("text", "")
                        if not text:
                            continue
                        # 如果当前文本长度够了，创建一个文本块
                        if count_tokens(text_buffer) > self.chunk_size:
                            chunk = Chunk(content=text_buffer, type=ChunkType.TEXT, metadata=item)
                            self._append_content_chunks(chunk)
                            text_buffer = ""

                        text_buffer += text + "\n"
                        last_page = page_idx
                    elif item["type"] == "image":
                        image_path = item.pop("img_path", None)
                        if not image_path:
                            continue
                        with myzip.open(image_path, "r") as f:
                            image_content = base64.b64encode(f.read()).decode("utf-8")
                        # FIXME: 下面这一行生成MIME类型的逻辑在某些文件类型下不一定正确，待验证
                        image_mime = f"image/{image_path.split('.')[-1]}"
                        # 将图片前后的文字内容作为上下文
                        image_context = self.find_item_context(content_list, idx)
                        img_desc = gen_image_desc(image_content, image_mime, image_context)
                        chunk = Chunk(content=img_desc, type=ChunkType.IMAGE, metadata={
                            "pages": item["pages"],
                            "image_base64": image_content,
                            "image_mime": image_mime,
                        })
                        self._append_content_chunks(chunk)
                    elif item["type"] == "table":
                        table_image_path = item.pop("img_path", None)
                        if not table_image_path:
                            continue
                        with myzip.open(table_image_path, "r") as f:
                            headers = {}
                            allow_redirects = True
                            if api_key := CONFIG["ocr_table"]["api_key"]:
                                headers["Authorization"] = f"Bearer {api_key}"
                                allow_redirects = False  # 远程调用禁止重定向
                            resp = requests.post(CONFIG["ocr_table"]["api_url"], headers=headers,
                                                 files={"files": f}, allow_redirects=allow_redirects)
                            parsed_tables = resp.json()
                            for x in parsed_tables:
                                chunk = Chunk(content=x, type=ChunkType.TABLE, metadata=item)
                                self._append_content_chunks(chunk)

                        with myzip.open(table_image_path, "r") as f:
                            # 把表格的图片也同时存储一份
                            table_image_content = base64.b64encode(f.read()).decode("utf-8")
                            table_image_mime = f"image/{table_image_path.split('.')[-1]}"
                            if "table_caption" in item:
                                table_image_context = item.pop("table_caption", "")
                            else:
                                # 将表格前后的文字内容作为上下文
                                table_image_context = self.find_item_context(content_list, idx)
                            img_desc = gen_image_desc(table_image_content, table_image_mime, table_image_context)
                            chunk = Chunk(content=img_desc, type=ChunkType.IMAGE, metadata={
                                "pages": item["pages"],
                                "image_base64": table_image_content,
                                "image_mime": table_image_mime,
                            })
                            self._append_content_chunks(chunk)
                    elif item["type"] == "equation":
                        equation_str = item["text"]
                        if len(text_buffer) > 0:
                            text_buffer += equation_str
                        else:
                            for prev_chunk in self.chunks.content_chunks[::-1]:
                                if prev_chunk.metadata["type"] == "text":
                                    prev_chunk.content += equation_str
                                    break
                except:
                    logger.exception(f"Error handling raw item {item} (doc {self.target.doc_id})")
            # 防止最后一页的内容被忽略
            if text_buffer:
                chunk = Chunk(
                    content=text_buffer,
                    type=ChunkType.TEXT,
                    metadata={"pages": str(int(last_page) + 1)}
                )
                self._append_content_chunks(chunk)

    def find_item_context(self, content_list, index, distance=3) -> str:
        buffer = []

        # 往前找上文
        for i in range(1, min(distance, index)):
            if content_list[index - i].get('type', '') == 'text':
                buffer.insert(0, content_list[index - i]['text'])
            # 这里我们假设图片前面的标题再前面的内容是与图片无关的，不适合作为上文使用。
            if "text_level" in content_list[index - i]:
                break

        # 往后找下文
        for i in range(1, min(distance, len(content_list) - index)):
            # 这里我们假设图片后面的标题及其再后面的内容都是与图片无关的，不适合作为下文使用。
            if "text_level" in content_list[index + i]:
                break
            if content_list[index + i].get('type', '') == 'text':
                buffer.append(content_list[index + i]['text'])

        return " ".join(buffer)

    def garbled_content_detect(self, file_path: str, block_size=20, threshold=0.2):
        with fitz.open(file_path) as doc:
            pages = []
            # 最多只获取前「block_size」页内容
            for i, page in enumerate(doc):
                if i >= block_size:
                    break
                pages.append(page)

            total = len(pages)
            # 20% 以上文档乱码即需要ocr栅格化
            garbled_limit = threshold * total
            for page in pages:
                # 普通页面进行文本抽取, 去除图片解析出的乱码
                page_content = clean_text(page.get_text().strip())
                if is_garbled(page_content):
                    garbled_limit -= 1
                    if garbled_limit <= 0:
                        return True
            return False

    def pdf_rasterized(self, file_path: str):
        new_path = file_path.rsplit('.', 1)[0] + '_rasterized.pdf'
        new_doc = fitz.open()
        # 将每一页转换为图片并添加到新文档
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                # 获取页面的像素图
                pix = page.get_pixmap()
                # 将像素图转换为PDF页面
                new_page = new_doc.new_page(width=pix.width, height=pix.height)
                new_page.insert_image(new_page.rect, pixmap=pix)
        new_doc.save(new_path)
        new_doc.close()
        logger.info(f"已生成栅格化文档: {new_path}")
        return new_path
