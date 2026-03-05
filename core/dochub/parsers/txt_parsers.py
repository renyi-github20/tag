from typing import Dict, Any

from chardet import UniversalDetector

from dochub.parsers.base import BaseDocumentParser
from dochub.schemas import Chunk, Param, DataType, ChunkType
from dochub.utils.text_splitters import RecursiveCharacterTextSplitter

from utils.i18n import I18NString, Language
from utils.token import count_tokens


def slices_generator(lines, slice_size):
    for i in range(0, len(lines), slice_size):
        yield lines[i:i + slice_size]


class GeneralTextualDocumentParser(BaseDocumentParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })

    target_content_type = "text/*"
    target_file_ext = "*"

    params = [
        Param(name="chunk_size", display_name="文本块大小", required=True, data_type=DataType.NUMBER, default=500),
        Param(name="chunk_overlap", display_name="文本块间重叠字符数", required=False, data_type=DataType.NUMBER,
              default=100),
    ]

    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.chunk_size = kwargs.get("chunk_size", 500)
        self.chunk_overlap = kwargs.get("chunk_overlap", 100)

    def _parse_impl(self) -> None:
        with open(self.target.physical_path, 'rb') as f:
            encoding_detect_res = detect_encoding(self.target.physical_path)

            if encoding_detect_res['encoding'] is None or encoding_detect_res['confidence'] < 0.7:
                raise ValueError(f"Document {self.target.doc_id} is likely a binary file, which should not be parsed.")

        with open(self.target.physical_path, "r", encoding=encoding_detect_res["encoding"], errors="ignore") as f:
            content = f.read()
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=500,
                chunk_overlap=100,
                length_function=count_tokens
            )
            snippets = splitter.split_text(content)
            for snippet in self._progress_wrapper(snippets, total=len(snippets)):
                chunk = Chunk(content=snippet, type=ChunkType.TEXT)
                self._append_content_chunks(chunk)

            self._set_fulltext_chunk(content)
            self._report_progress(100)


class GeneralTxtParser(GeneralTextualDocumentParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })

    target_content_type = "text/plain"
    target_file_ext = "txt"


def detect_encoding(file_path: str) -> Dict[str, Any]:
    detector = UniversalDetector()
    with open(file_path, "rb") as f:
        for line in f.readlines():
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result