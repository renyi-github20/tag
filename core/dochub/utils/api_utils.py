import os
from typing import List, Dict, Any

import requests
from loguru import logger

from config import CONFIG
from dochub.schemas import Document
from utils import storage_utils
from utils.token import truncate

session = requests.Session()


def get_text_embedding(data, batch_size=1000):
    if isinstance(data, str):
        data = [data]
    assert len(data) > 0, f'文本列表不能为空'

    data_embeddings = []
    headers = {}
    if api_key := CONFIG["embedder"]["api_key"]:
        headers["Authorization"] = f"Bearer {api_key}"
    for i in range(0, len(data), batch_size):
        resp_data = session.post(
            CONFIG["embedder"]["api_url"],
            headers=headers,
            json={
                "input": data[i:i + batch_size],
            }
        ).json()["data"]
        data_embeddings.extend(e["embedding"] for e in resp_data)
    assert len(data) == len(data_embeddings), f'文本和向量数据不对应, 文本{len(data)}, 向量{len(data_embeddings)}'
    # logger.info(
    #     f'data2vector done! 文本数据：{len(data_embeddings)}, 向量数据：{len(data)}, 向量维度: {len(data_embeddings[0])}')

    return data_embeddings


storage_utils.create_dir_if_not_exists("/tmp/docx2pdf")


def preview_docx(doc: Document) -> tuple[bytes, str] | None:
    file_path = os.path.join("/tmp/docx2pdf", doc.doc_id + ".pdf")
    if not os.path.exists(file_path):
        url = CONFIG["pdf_converter"]["api_url"]
        with open(doc.physical_path, "rb") as src_file:
            files = {'file': (doc.doc_name, src_file)}
            resp = requests.post(url, files=files)
        if resp.status_code == 200:
            # 保存文件
            with open(file_path, mode="wb+") as out_file:
                out_file.write(resp.content)
            return resp.content, file_path
        else:
            logger.exception(f"pdf-converter response with code {resp.status_code}", resp.json())
            return None
    else:
        with open(file_path, mode="rb") as in_file:
            return in_file.read(), file_path


class AudioContentProcessor:
    @staticmethod
    def merge_segments(segments: List[Dict[str, Any]], lang: str):
        content = ""
        for segment in segments:
            punc = ". "
            if "zh" in lang:
                punc = "。"
            if "ja" in lang:
                punc = "、"
            content += segment["text"]
            if segment["text"][-1] not in [".", "。", "，", ",", "、", " ", "　"]:
                content += punc
        return content


def get_audio_transcriptions(file_path: str):
    params = {
        "model": CONFIG["asr"]["model"],
        "response_format": "verbose_json",
        "stream": False
    }
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        headers = {}
        if api_key := CONFIG["asr"]["api_key"]:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(CONFIG["asr"]["api_url"], files=files, data=params,
                                 headers=headers).json()

        content = AudioContentProcessor.merge_segments(response["segments"], response["language"])
        segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
            for segment in response["segments"]
        ]
        return {
            "content": content,
            "language": response["language"],
            "duration": response["duration"],
            "segments": segments,
        }


summarize_system_prompt = "The assistant always gives an accurate summary over the document text given by the user. The assistant is multilingual and outputs in the same language as the given document text uses."


def text_summary(text: str) -> str:
    text = truncate(text, CONFIG["summary"]["context_length"])
    headers = {}
    if api_key := CONFIG["summary"]["api_key"]:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": CONFIG["summary"]["model"],
        "messages": [
            {"role": "system", "content": summarize_system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": CONFIG["summary"]["max_tokens"],
        "stream": False,
    }
    resp = requests.post(CONFIG["summary"]["api_url"], headers=headers, json=payload).json()
    return resp["choices"][0]["message"]["content"]


def video_summary(video_url: str, query: str = "请总结视频内容（1000 字左右）") -> str:
    headers = {}
    if api_key := CONFIG["vl"]["api_key"]:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": CONFIG["vl"]["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": query}
                ]
            }
        ],
        "temperature": 0.2,
        "top_k": 30,
        "max_tokens": CONFIG["vl"]["max_tokens"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stream": False
    }
    resp = requests.post(CONFIG["vl"]["api_url"], headers=headers, json=payload).json()
    return resp["choices"][0]["message"]["content"]


image_desc_prompt = "如果图片上有文字，将文字按原有的格式输出成文本或者图表格式，并对图片进行简短概括。如果是图片型图片，对图片进行详细描述。"


def gen_image_desc(base64_str: str, mime: str="image/png", caption: str = None) -> str:
    headers = {}
    if api_key := CONFIG["vl"]["api_key"]:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": CONFIG["vl"]["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (f"结合图片给定的上下文信息“{caption}”，" if caption else "") + image_desc_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_str}"
                        }
                    }
                ]
            }
        ],
        "temperature": 1,
        "top_p": 0.7,
        "max_tokens": CONFIG["vl"]["max_tokens"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stream": False,
    }
    resp = requests.post(CONFIG["vl"]["api_url"], headers=headers, json=payload).json()
    return resp["choices"][0]["message"]["content"]
