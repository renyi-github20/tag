import copy
from abc import ABC

from dochub.parsers.base import BaseDocumentParser
from dochub.parsers.pdf_parsers import GeneralPDFParser
from dochub.utils import api_utils
from utils.i18n import I18NString, Language


class BaseDocParser(BaseDocumentParser, ABC):
    target_content_type = [
        "application/msword",
        "application/wps-office.doc"
    ]
    target_file_ext = "doc"


class GeneralDocParser(BaseDocParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })

    def _parse_impl(self) -> None:
        # doc文档需要先转成pdf然后再用pdf parser解析
        ret = api_utils.preview_docx(self.target)
        if ret is None:
            raise Exception("doc文档解析失败")
        doc_cpy = copy.deepcopy(self.target)
        doc_cpy.physical_path = ret[1]
        self.chunks = GeneralPDFParser(doc_cpy, **self.kwargs).parse()
        return
