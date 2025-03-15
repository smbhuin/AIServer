from typing import List
from api_types import TranscriptionSegment
from PIL.Image import Image
from io import BytesIO
import base64
import uuid

def to_timestamp(t: int, separator=',') -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"

def vtt_from(segments: List[TranscriptionSegment]) -> str:
    data = "WEBVTT\n\n"
    for seg in segments:
        data += f"{to_timestamp(seg.start, separator='.')} --> {to_timestamp(seg.end, separator='.')}\n"
        data += f"{seg.text}\n\n"
    return data

def srt_from(segments: List[TranscriptionSegment]) -> str:
    data = ""
    for i in range(len(segments)):
        seg = segments[i]
        data += f"{i+1}\n"
        data += f"{to_timestamp(seg.start, separator=',')} --> {to_timestamp(seg.end, separator=',')}\n"
        data += f"{seg.text}\n\n"
    return data

def text_from(segments: List[TranscriptionSegment]) -> str:
    return "\n".join(seg.text for seg in segments)

def b64_str_from(image: Image) -> str:
    with BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        b64_data = base64.b64encode(output_bytes.getvalue())
    return b64_data.decode()

def save_image(image: Image, path: str) -> str:
    file_name = f"{uuid.uuid4()}.png"
    image.save(f"{path}/{file_name}")
    return file_name
