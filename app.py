import dash
from flask import Response

import cv2
import depthai as dai # this is the oak-d-lite computer

from dash import html

def simple_layout():
    return html.Div([
            html.Iframe(src="/video_feed", width="1200", height="800", id="video-stream"),
            ], style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
            )

app = dash.Dash(__name__)
app.layout = simple_layout()

def generate():
    pipeline = dai.Pipeline()

    # Create the ColorCamera node and set its properties
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Create the XLinkOut node for the video stream and set its properties
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("My first stream")

    # Link the ColorCamera to the XLinkOut node
    camRgb.video.link(xoutRgb.input)

    # Start the pipeline
    with dai.Device(pipeline) as device:
        video_queue = device.getOutputQueue(name="My first stream", maxSize=4, blocking=False) # get the video stream queue
        
        while True:
            frame = video_queue.get().getCvFrame() # get the video frame

          # <you can add fancy processing of the video frame here>

            (flag, encodedImage) = cv2.imencode(".jpg", frame) # encode the frame into a jpeg image
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.server.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)