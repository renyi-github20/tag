import base64
import gc
import hashlib
import itertools
import math
import os
import shutil
from functools import cache
from io import BytesIO
from typing import Union, BinaryIO

import av
from loguru import logger
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


@cache
def get_hashed_name(video_file_path):
    video_filename = os.path.basename(video_file_path)
    return hashlib.md5(video_filename.encode()).hexdigest()


def get_video_info(video_path):
    """
    使用 PyAV 获取视频的时长和帧率。
    """
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream_duration = stream.duration
            if not stream_duration:
                stream_duration = stream.container.duration / 1000.0
            duration = float(stream_duration * stream.time_base)
            frame_rate = stream.average_rate
            if not frame_rate:
                frame_rate = 1.0
            return duration, frame_rate
    except Exception as e:
        print(f"Error opening video with PyAV: {e}")
        return None, None


def images_to_video(image_files, output_video_path, fps=1):
    """
    将指定文件夹中的图片转换为视频。

    Args:
        image_files:  有序图片路径列表。
        output_video_path: 输出视频的路径 (例如: output.mp4)。
        fps:  帧率，默认为 1。
    """

    try:
        with av.open(image_files[0]) as img_container:
            first_image = next(img_container.decode(video=0))
            width, height = first_image.width, first_image.height

        with av.open(output_video_path, mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for image_file in image_files:
                with av.open(image_file) as img_container:
                    for frame in img_container.decode(video=0):
                        for packet in stream.encode(frame):
                            container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        logger.info(f"视频已成功创建：{output_video_path}")

    except FileNotFoundError:
        logger.error("错误：未找到图片文件。")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


def safe_scale(original_size, max_size):
    o_width, o_height = original_size
    max_width, max_height = max_size
    if o_width < o_height:
        max_width, max_height = max_height, o_height
    scale_w = math.floor(o_width * min(max_width / o_width, max_height / o_height) / 2.) * 2
    scale_h = math.floor(o_height * min(max_width / o_width, max_height / o_height) / 2.) * 2
    return scale_w, scale_h


def extract_frames(video_path, output_base_dir="tmp"):
    """
    抽取视频关键帧并进行去重。使用 PyAV。

    Args:
        video_path: 视频文件路径。
        output_base_dir: 输出目录的基目录，默认为 "tmp"。
    """
    duration, frame_rate = get_video_info(video_path)
    if duration is None or frame_rate is None:
        return

    # 计算均匀帧索引，当 skip_frame 失效时使用
    backup_frame_idx = [i * frame_rate for i in range(int(duration * frame_rate))]

    hashed_name = get_hashed_name(video_path)
    output_dir = os.path.join(output_base_dir, hashed_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Extracting frames from: {video_path}")

    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            # Determine scaling dimensions
            width = stream.width
            height = stream.height
            scale_w, scale_h = safe_scale((width, height), (420, 360))

            extracted_frames_pil = []
            extracted_frames = []

            for i, frame in enumerate(container.decode(video=0)):
                # 部分视频下 skip_frame 不会成功生效，需要手动跳过非关键帧加速
                # 同时避免短视频关键帧过少，部分 fallback 到均匀截图模式工作
                if not frame.key_frame and i not in backup_frame_idx:
                    continue
                frame = frame.reformat(int(scale_w), int(scale_h))  # Rescale the frame.
                frame_idx = int(frame.time * 100)
                frame_path = os.path.join(output_dir, f"{hashed_name}-{frame_idx:08d}.png")
                extracted_frames_pil.append((frame.to_image(), frame_path))
                extracted_frames.append(f"{hashed_name}-{frame_idx:08d}.png")

            for frame, frame_path in tqdm(extracted_frames_pil):
                frame.save(frame_path)

        logger.info(f"Extracting frames: {len(extracted_frames)}")

        # SSIM-based deduplication
        selected_frames = [extracted_frames[0]]
        last_selected_frame = io.imread(os.path.join(output_dir, extracted_frames[0]))
        last_selected_frame = img_as_float(last_selected_frame)

        for frame_name in extracted_frames[1:]:
            frame_path = os.path.join(output_dir, frame_name)
            current_frame = io.imread(frame_path)
            current_frame = img_as_float(current_frame)

            similarity = ssim(last_selected_frame, current_frame, multichannel=True, channel_axis=-1, data_range=1.0)

            if similarity < 0.7:
                selected_frames.append(frame_name)
                last_selected_frame = current_frame

        # Delete unselected frames
        for frame_name in extracted_frames:
            if frame_name not in selected_frames:
                os.remove(os.path.join(output_dir, frame_name))

        return output_dir

    except Exception as e:
        logger.exception(f"Error during frame extraction with PyAV: {e}")
        return None


def capture_thumbnail_image(video_path, max_size: int = 500):
    """
    抽取视频缩略图。

    Args:
        video_path: 视频文件路径。
        max_size: 截图的最大尺寸，默认为 500，表示不超过 500x500px。
    """

    print(f"Capturing thumbnail from: {video_path}")
    thumbnail = None
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            # Determine scaling dimensions
            width = stream.width
            height = stream.height
            scale_f = min(max_size / width, max_size / height)

            for i, frame in tqdm(enumerate(container.decode(video=0))):
                if not frame.key_frame:
                    continue
                if i < 5:
                    if thumbnail is None:
                        frame = frame.reformat(int(width * scale_f), int(height * scale_f))
                        thumbnail = frame.to_image()
                    continue
                frame = frame.reformat(int(width * scale_f), int(height * scale_f))
                thumbnail = frame.to_image()
                break

        bytes_buffer = BytesIO()
        thumbnail.save(bytes_buffer, format="JPEG")
        bytes_buffer.seek(0)
        compressed_image = bytes_buffer.read()
        bytes_buffer.close()
        return base64.b64encode(compressed_image).decode("utf-8")
    except Exception as e:
        logger.exception(f"Error during frame extraction with PyAV: {e}")
        return None


def extract_into_thumbnail_video(video_in_path, output_base_dir="tmp", max_frames=100):
    output_dir = extract_frames(video_in_path, output_base_dir)
    if output_dir:  # Check if extract_frames was successful
        os.makedirs(output_dir + "-vid", exist_ok=True)
        # 切分成 max_frame 长度的片段
        image_files = []
        for filename in os.listdir(output_dir):
            if filename.lower().endswith((".png")):
                image_files.append(os.path.join(output_dir, filename))
        image_files.sort()
        if not image_files:
            logger.exception("错误：文件夹中没有找到图片文件。")
            return []
        process_list = [image_files[i:i + max_frames] for i in range(0, len(image_files), max_frames)]
        # 图片切片转视频
        output_path_list = []
        for idx, job in enumerate(process_list):
            start_pts = int(job[0][-12:][:-4]) / 100.0
            end_pts = int(job[-1][-12:][:-4]) / 100.0
            output_path = os.path.join(output_dir + "-vid", f"segment_{idx:04d}.mp4")
            images_to_video(job, output_path)
            output_path_list.append((output_path, start_pts, end_pts))
        # 删除截图目录
        shutil.rmtree(output_dir)
        return output_path_list
    else:
        return []


def extract_and_convert_audio(
        input_file: Union[str, BinaryIO],
        output_file: str = None,
        sampling_rate: int = 16000,
        split_stereo: bool = False,
):

    with av.open(output_file, mode="w") as output_container:
        stream = output_container.add_stream("aac", rate=16000, layout="mono")

        with av.open(input_file, mode="r", metadata_errors="ignore") as input_container:
            if len(input_container.streams.audio) == 0:
                return False
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono" if not split_stereo else "stereo",
                rate=sampling_rate,
            )
            frames = input_container.decode(audio=0)
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, 500000)
            frames = _resample_frames(frames, resampler)

            # 遍历输入文件的音频帧并写入输出文件
            for frame in frames:
                frame.pts = None
                for packet in stream.encode(frame):
                    output_container.mux(packet)

            # 写入剩余数据
            for packet in stream.encode(None):
                output_container.mux(packet)

    del resampler
    gc.collect()
    return True


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def encode_into_base64(file_path: str):
    return "data:video/mp4;base64," + base64.b64encode(open(file_path, "rb").read()).decode("utf-8")
