import streamlit as st
import wave

# from ASR import recognize
import requests
import pyaudio
import numpy as np
import base64
import io
import os
import time
import tempfile
import librosa
import traceback
from pydub import AudioSegment
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions


# Định nghĩa URL API
API_URL = os.getenv("API_URL", "http://127.0.0.1:60808/chat")

# Các tham số ghi âm
IN_FORMAT = pyaudio.paInt16
IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

# Các tham số phát âm
OUT_FORMAT = pyaudio.paInt16
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760


# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def run_vad(ori_audio, sr):
    """
    Thực hiện Voice Activity Detection (VAD) trên audio đầu vào
    """
    _st = time.time()
    try:
        # Chuyển đổi audio thành mảng numpy
        audio = np.frombuffer(ori_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        # Resample audio nếu cần
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        # Thực hiện VAD
        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        # Chuyển đổi lại về định dạng ban đầu
        if sr != sampling_rate:
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()

        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        # Xử lý lỗi nếu có
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


def warm_up():
    """
    Khởi động VAD để giảm thời gian xử lý lần đầu
    """
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


def save_tmp_audio(audio_bytes):
    """
    Lưu audio tạm thời vào file
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        file_name = tmpfile.name
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=OUT_SAMPLE_WIDTH,
            frame_rate=OUT_RATE,
            channels=OUT_CHANNELS,
        )
        audio.export(file_name, format="wav")
        return file_name


def speaking(status):
    """
    Xử lý và phát âm thanh phản hồi
    """

    # Khởi tạo PyAudio
    p = pyaudio.PyAudio()

    # Mở stream PyAudio
    stream = p.open(
        format=OUT_FORMAT, channels=OUT_CHANNELS, rate=OUT_RATE, output=True
    )

    # Chuẩn bị audio buffer
    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, "wb")
    wf.setnchannels(IN_CHANNELS)
    wf.setsampwidth(IN_SAMPLE_WIDTH)
    wf.setframerate(IN_RATE)
    total_frames = b"".join(st.session_state.frames)
    dur = len(total_frames) / (IN_RATE * IN_CHANNELS * IN_SAMPLE_WIDTH)
    status.warning(f"Speaking... recorded audio duration: {dur:.3f} s")
    wf.writeframes(total_frames)

    # Lưu và hiển thị audio đầu vào
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with open(tmpfile.name, "wb") as f:
            f.write(audio_buffer.getvalue())
        file_name = tmpfile.name
        with st.chat_message("user"):
            st.audio(file_name, format="audio/wav", loop=False, autoplay=False)
        st.session_state.messages.append(
            {"role": "assistant", "content": file_name, "type": "audio"}
        )

    st.session_state.frames = []

    # Gửi yêu cầu đến API và xử lý phản hồi
    audio_bytes = audio_buffer.getvalue()
    base64_encoded = str(base64.b64encode(audio_bytes), encoding="utf-8")
    files = {"audio": base64_encoded}
    output_audio_bytes = b""
    with requests.post(API_URL, json=files, stream=True) as response:
        try:
            for chunk in response.iter_content(chunk_size=OUT_CHUNK):
                if chunk:
                    # Chuyển đổi chunk thành mảng numpy
                    output_audio_bytes += chunk
                    audio_data = np.frombuffer(chunk, dtype=np.int8)
                    # Phát audio
                    stream.write(audio_data)
        except Exception as e:
            st.error(f"Error during audio streaming: {e}")

    # Lưu và hiển thị audio đầu ra
    out_file = save_tmp_audio(output_audio_bytes)
    with st.chat_message("assistant"):
        st.audio(out_file, format="audio/wav", loop=False, autoplay=False)
    st.session_state.messages.append(
        {"role": "assistant", "content": out_file, "type": "audio"}
    )

    # Đóng các stream và kết thúc
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.session_state.speaking = False
    st.session_state.recording = True


def recording(status):
    """
    Ghi âm từ microphone
    """
    audio = pyaudio.PyAudio()

    # Mở stream audio đầu vào
    stream = audio.open(
        format=IN_FORMAT,
        channels=IN_CHANNELS,
        rate=IN_RATE,
        input=True,
        frames_per_buffer=IN_CHUNK,
    )

    temp_audio = b""
    vad_audio = b""

    start_talking = False
    last_temp_audio = None
    st.session_state.frames = []

    while st.session_state.recording:
        status.success("Listening...")
        audio_bytes = stream.read(IN_CHUNK)
        temp_audio += audio_bytes

        # Thực hiện VAD khi đủ độ dài audio
        if len(temp_audio) > IN_SAMPLE_WIDTH * IN_RATE * IN_CHANNELS * VAD_STRIDE:
            dur_vad, vad_audio_bytes, time_vad = run_vad(temp_audio, IN_RATE)

            print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

            # Xử lý kết quả VAD
            if dur_vad > 0.2 and not start_talking:
                if last_temp_audio is not None:
                    st.session_state.frames.append(last_temp_audio)
                start_talking = True
            if start_talking:
                st.session_state.frames.append(temp_audio)
            if dur_vad < 0.1 and start_talking:
                st.session_state.recording = False
                print(f"speech end detected. excit")
            last_temp_audio = temp_audio
            temp_audio = b""

    # Đóng stream và kết thúc
    stream.stop_stream()
    stream.close()
    audio.terminate()


def main():
    """
    Hàm chính của ứng dụng
    """

    st.title("Chat Mini-Omni Demo")
    status = st.empty()

    # Khởi tạo các biến trạng thái
    if "warm_up" not in st.session_state:
        warm_up()
        st.session_state.warm_up = True
    if "start" not in st.session_state:
        st.session_state.start = False
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "speaking" not in st.session_state:
        st.session_state.speaking = False
    if "frames" not in st.session_state:
        st.session_state.frames = []

    if not st.session_state.start:
        status.warning("Click Start to chat")

    # Tạo nút Start
    start_col, stop_col, _ = st.columns([0.2, 0.2, 0.6])
    start_button = start_col.button("Start", key="start_button")
    if start_button:
        time.sleep(1)
        st.session_state.recording = True
        st.session_state.start = True

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "msg":
                st.markdown(message["content"])
            elif message["type"] == "img":
                st.image(message["content"], width=300)
            elif message["type"] == "audio":
                st.audio(
                    message["content"], format="audio/wav", loop=False, autoplay=False
                )

    # Vòng lặp chính của ứng dụng
    while st.session_state.start:
        if st.session_state.recording:
            recording(status)

        if not st.session_state.recording and st.session_state.start:
            st.session_state.speaking = True
            speaking(status)


if __name__ == "__main__":
    main()
