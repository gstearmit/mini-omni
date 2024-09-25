"""A simple web interactive chat demo based on gradio."""

import os
import time
import gradio as gr
import base64
import numpy as np
import requests


API_URL = os.getenv("API_URL", None)
client = None

if API_URL is None:
    from inference import OmniInference
    omni_client = OmniInference('./checkpoint', 'cuda:0')
    omni_client.warm_up()


OUT_CHUNK = 4096
OUT_RATE = 24000
OUT_CHANNELS = 1


# Hàm xử lý audio đầu vào
def process_audio(audio):
    filepath = audio
    print(f"filepath: {filepath}")
    if filepath is None:
        return

    cnt = 0
    if API_URL is not None:
        # Nếu có API_URL, gửi yêu cầu đến server
        with open(filepath, "rb") as f:
            data = f.read()
            # Mã hóa dữ liệu audio thành base64
            base64_encoded = str(base64.b64encode(data), encoding="utf-8")
            files = {"audio": base64_encoded}
            tik = time.time()
            # Gửi POST request đến API
            with requests.post(API_URL, json=files, stream=True) as response:
                try:
                    # Xử lý từng chunk dữ liệu nhận được
                    for chunk in response.iter_content(chunk_size=OUT_CHUNK):
                        if chunk:
                            # Chuyển đổi chunk thành numpy array
                            if cnt == 0:
                                print(f"first chunk time cost: {time.time() - tik:.3f}")
                            cnt += 1
                            audio_data = np.frombuffer(chunk, dtype=np.int16)
                            audio_data = audio_data.reshape(-1, OUT_CHANNELS)
                            yield OUT_RATE, audio_data.astype(np.int16)

                except Exception as e:
                    print(f"error: {e}")
    else:
        # Nếu không có API_URL, sử dụng OmniInference trực tiếp
        tik = time.time()
        for chunk in omni_client.run_AT_batch_stream(filepath):
            # Chuyển đổi chunk thành numpy array
            if cnt == 0:
                print(f"first chunk time cost: {time.time() - tik:.3f}")
            cnt += 1
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_data = audio_data.reshape(-1, OUT_CHANNELS)
            yield OUT_RATE, audio_data.astype(np.int16)


# Hàm chính để khởi chạy giao diện Gradio
def main(port=None):

    # Tạo giao diện Gradio
    demo = gr.Interface(
        process_audio,
        inputs=gr.Audio(type="filepath", label="Microphone"),
        outputs=[gr.Audio(label="Response", streaming=True, autoplay=True)],
        title="Chat Mini-Omni Demo",
        live=True,
    )
    if port is not None:
        # Khởi chạy demo với port cụ thể
        demo.queue().launch(share=False, server_name="0.0.0.0", server_port=port)
    else:
        # Khởi chạy demo với port mặc định
        demo.queue().launch()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
