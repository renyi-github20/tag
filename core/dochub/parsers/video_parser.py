import os
from abc import ABC

from config import CONFIG
from dochub.parsers.base import BaseDocumentParser
from dochub.schemas import Chunk, DataType, Param, ChunkType
from dochub.utils.api_utils import video_summary, get_audio_transcriptions
from dochub.utils.av_utils import extract_and_convert_audio, extract_into_thumbnail_video, encode_into_base64, \
    capture_thumbnail_image
from utils.i18n import I18NString, Language


summary_prompt = I18NString({
    Language.ZH: "请结合以下 ASR 转录文本，对视频内容进行总结（1000字左右）。请使用源文本中主要使用的语言进行总结。\n转录文本：{thumbnail_text}\n",
    Language.EN: "Please summarize the video content (around 1000 words) based on the following ASR transcription text. Use the main language used in the source text for summarization.\nTranscription text: {thumbnail_text}\n"
})

class BaseVideoParser(BaseDocumentParser, ABC):
    target_content_type = ["video"]
    target_file_ext = ["mp4", "mkv", "webm", "gif", "avi", "mov"]


class GeneralVideoParser(BaseVideoParser, ABC):
    name = I18NString({
        Language.ZH: "通用",
        Language.EN: "General",
    })

    params = [
        Param(name="vl_enable", display_name="开启VL解析", required=True, data_type=DataType.BOOLEAN, default=False),
    ]

    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.vl_enable = kwargs.get("vl_enable", False)

    def _parse_impl(self) -> None:
        self._report_progress(1)

        # 音视频格式转换为m4a
        audio_path = self.target.physical_path
        target_path = os.path.join(self._get_tmp_data_path(), os.path.basename(audio_path).split(".")[0] + ".m4a")
        has_audio = extract_and_convert_audio(audio_path, target_path)

        video_cover = capture_thumbnail_image(self.target.physical_path)

        if has_audio:
            transcriptions = get_audio_transcriptions(target_path)
            chunk = Chunk(content=transcriptions["content"],
                          type=ChunkType.TEXT, metadata={"content_source": "asr"})
            self._append_content_chunks(chunk)
            self._set_attribute_chunk(metadata={
                              "language": transcriptions["language"],
                              "duration": transcriptions["duration"],
                              "segments": transcriptions["segments"],
                              "video_cover": video_cover
                          })
            self.asr_segments = transcriptions["segments"]
            self._report_progress(50.0)

        # 2. 尝试调用 VL 处理
        if not self.vl_enable:
            self._report_progress(100)
            return

        video_path = self.target.physical_path
        output_base_dir = os.path.join(self._get_tmp_data_path(), "thumbnail")
        thumbnail_video_list = extract_into_thumbnail_video(
            video_path, output_base_dir, max_frames=CONFIG["vl"]["max_frames"])

        # 2.1 ASR文本切片对应到视频切片片段
        asr_text_list = [""] * len(thumbnail_video_list)
        if has_audio and len(self.asr_segments) > 0:
            asr_text_list = []
            for thumbnail_video in thumbnail_video_list:
                thumbnail_video_asr = ""
                for segment in self.asr_segments:
                    if segment.get("start", 0) < thumbnail_video[1] or segment.get("end", 0) > thumbnail_video[2]:
                        continue
                    else:
                        thumbnail_video_asr += segment.get("text", "")
                asr_text_list.append(thumbnail_video_asr)

        # 2.2 VL解析视频并总结
        for idx, (thumbnail_text, thumbnail_info) in enumerate(zip(asr_text_list, thumbnail_video_list)):
            # 如果存在 ASR
            if thumbnail_text:
                summary = video_summary(
                    encode_into_base64(thumbnail_info[0]),
                    query=summary_prompt.format(thumbnail_text),
                )
            else:
                summary = video_summary(encode_into_base64(thumbnail_info[0]))
            self._append_content_chunks(Chunk(
                content=summary,
                type=ChunkType.TEXT,
                metadata={
                    "content_source": "video_summary",
                    "start": thumbnail_info[1],  # 单位是秒
                    "end": thumbnail_info[2],
                }
            ))
            self._report_progress(50.0 + 49.0 * (idx + 1) / len(thumbnail_video_list))
        self._report_progress(100)
