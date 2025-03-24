import cv2
import depthai as dai
from flask import Flask, Response, render_template

app = Flask(__name__)

def generate(camera_id):
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    
    if camera_id == 1:
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    else:
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName(f"video{camera_id}")
    camRgb.video.link(xoutRgb.input)

    with dai.Device(pipeline) as device:
        video_queue = device.getOutputQueue(name=f"video{camera_id}", maxSize=4, blocking=False)
        
        while True:
            frame = video_queue.get().getCvFrame()
            if frame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    return Response(generate(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)