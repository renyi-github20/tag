import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from loguru import logger

import config
from dochub.schemas import Chunk, Document, Param, DocumentChunks, ChunkType
from dochub.utils import progress_reporter
from dochub.utils.api_utils import text_summary
from utils.i18n import I18NString


class BaseDocumentParser(ABC):
    name: I18NString
    target_content_type: str | List[str]
    target_file_ext: str | List[str]
    params: List[Param] = []
    parse_progress_slices = random.sample([7, 11, 13, 17], 1)[0]

    def __init__(self, target: Document, **kwargs):
        # self._validate_document()
        self.target = target
        self.kwargs = kwargs
        self.chunks: DocumentChunks = DocumentChunks()

    def parse(self) -> DocumentChunks:
        self._parse_impl()
        # 如果未生成fulltext Chunk，则从content_chunks拼接fulltext
        if not self.chunks.fulltext_chunk:
            fulltext = "\n".join([chunk.content for chunk in self.chunks.content_chunks])
            self._set_fulltext_chunk(fulltext)
        summary = text_summary(self.chunks.fulltext_chunk.content)
        self._set_summary_chunk(summary)
        return self.chunks

    def _validate_document(self):
        if isinstance(self.target_content_type, str):
            assert self.target_content_type == self.target.content_type, (
                "文档类型不匹配！期望的文档类型：{}，得到的文档类型：{}".format(
                self.target_content_type, self.target.content_type)
            )
        else:
            assert self.target.content_type in self.target_content_type, (
                "文档类型不匹配！期望的文档类型：{}，得到的文档类型：{}"
                .format(self.target_content_type, self.target.content_type)
            )

    def _append_content_chunks(self, *chunks: Chunk):
        self.chunks.content_chunks.extend(chunks)

    def _set_attribute_chunk(self, metadata: Dict[str, Any]):
        self.chunks.attribute_chunk = Chunk(content="", type=ChunkType.ATTRIBUTE, metadata=metadata)

    def _set_fulltext_chunk(self, fulltext: str):
        self.chunks.fulltext_chunk = Chunk(content=fulltext, type=ChunkType.FULLTEXT)

    def _set_summary_chunk(self, summary: str):
        self.chunks.summary_chunk = Chunk(content=summary, type=ChunkType.SUMMARY)

    def _get_tmp_data_path(self):
        path = os.path.join(config.data_dir, "doc_parsers", self.target.doc_id)
        os.makedirs(path, exist_ok=True)
        return path

    @abstractmethod
    def _parse_impl(self) -> None:
        pass

    def _report_progress(self, percent: float):
        try:
            progress_reporter.report_progress(self.target.doc_id, percent)
        except:
            logger.exception(f"Failed to report document parsing progress: {percent:2f} ({self.target.doc_id})")

    def _progress_wrapper(self, generator, total, start_percent=0, end_percent=100):
        parse_unit = max(total // self.parse_progress_slices, 1)
        for i, v in enumerate(generator):
            yield v

            if i % parse_unit == 0:
                percent = ((i + 1) / total) * (end_percent - start_percent) + start_percent
                self._report_progress(percent)
