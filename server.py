import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
from inference import OmniInference


# Lớp OmniChatServer để xử lý các yêu cầu chat
class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        # Khởi tạo Flask server
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False

        # Khởi tạo client OmniInference
        self.client = OmniInference(ckpt_dir, device)
        self.client.warm_up()

        # Đăng ký route /chat
        server.route("/chat", methods=["POST"])(self.chat)

        # Chạy server nếu run_app=True
        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    # Hàm xử lý yêu cầu chat
    def chat(self) -> Response:
        # Lấy dữ liệu từ request
        req_data = flask.request.get_json()
        try:
            # Giải mã audio từ base64
            data_buf = req_data["audio"].encode("utf-8")
            data_buf = base64.b64decode(data_buf)
            
            # Lấy các tham số
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            # Lưu audio tạm thời và xử lý
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                # Tạo audio generator
                audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
                # Trả về response dạng stream
                return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            # In ra lỗi nếu có
            print(traceback.format_exc())


# Hàm tạo app cho gunicorn
# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)
    return server.server


# Hàm chạy server
def serve(ip='0.0.0.0', port=60808, device='cuda:0'):
    OmniChatServer(ip, port=port,run_app=True, device=device)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)
    
