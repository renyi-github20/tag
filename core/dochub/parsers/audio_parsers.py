from abc import ABC

from dochub.parsers.base import BaseDocumentParser
from dochub.schemas import Chunk, ChunkType

from dochub.utils.api_utils import get_audio_transcriptions
from utils.i18n import I18NString, Language


class BaseAudioParser(BaseDocumentParser, ABC):
    target_content_type = ["audio", "video"]
    target_file_ext = ["mp3", "m4a", "wav", "weba", "flac", "ogg", "mp4", "avi", "mkv", "mov", "webm"]


class GeneralAudioParser(BaseAudioParser):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })

    # 当前未验证
    def _validate_document(self):
        content_type = self.target.content_type
        is_audio_or_video = True if "audio" in content_type or "video" in content_type else False
        assert is_audio_or_video, "媒体类型不匹配！期望的媒体类型：{}，得到的媒体类型：{}".format(
            self.target_content_type, self.target.content_type)

    def _parse_impl(self) -> None:
        self._report_progress(1)

        audio_path = self.target.physical_path
        try:
            transcriptions = get_audio_transcriptions(audio_path)
            self._append_content_chunks(Chunk(content=transcriptions["content"], type=ChunkType.TEXT))
            self._set_fulltext_chunk(transcriptions["content"])
            self._set_attribute_chunk(metadata={
                          "language": transcriptions["language"],
                          "duration": transcriptions["duration"],
                          "segments": transcriptions["segments"]
                      })
        except:
            raise Exception("asr run error for the file {} with doc id {}".format(audio_path, self.target.doc_id))

        self._report_progress(100)

