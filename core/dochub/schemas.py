from datetime import datetime
from enum import Enum, StrEnum
from typing import List, Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    PARSED = "parsed"
    SAVED = "saved"
    READY = "ready"
    ERROR = "error"


class Document(BaseModel):
    doc_id: str
    doc_name: str
    content_type: str
    physical_path: str
    logical_path: str | None = None
    doc_class_id: str | None = None
    metadata: Dict[str, Any] | None = None
    status: DocumentStatus = DocumentStatus.CREATED
    status_progress: float = 1.0
    created_time: datetime | None = Field(default_factory=datetime.now)
    creator: str | None = None


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FULLTEXT = "fulltext"
    IMAGE = "image"
    SQL = "sql"
    ATTRIBUTE = "attribute"  # 记录文档文本内容外的文档各种属性，如image/video的cover、video/audio的segments等
    SUMMARY = "summary"


class Chunk(BaseModel):
    content: str
    type: ChunkType
    metadata: Dict[str, Any] = {}


class DocumentChunks(BaseModel):
    content_chunks: List[Chunk] = []  # 包含 text、table、image Chunk
    fulltext_chunk: Optional[Chunk] = None # 包含文档解析出的文字全文内容
    attribute_chunk: Optional[Chunk] = None # 记录文档文本内容外的文档各种属性，如image/video的cover、video/audio的segments等
    summary_chunk: Optional[Chunk] = None


class DataType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class Param(BaseModel):
    name: str
    display_name: str
    required: bool
    data_type: DataType
    default: Optional[Any]


class DocumentParser(BaseModel):
    key: str
    name: str
    target_content_type: str | List[str]
    target_file_ext: str | List[str]
    params: List[Param]
