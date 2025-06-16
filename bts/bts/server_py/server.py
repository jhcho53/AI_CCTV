import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

#프레임 처리
from flask import Flask, jsonify, Response, send_file
import numpy as np
import cv2


#위는 원래 코드

context = 'Invasion'
description = "The children break in over the fence"
mp4 = None


def send_message(context, description):


    #Firebase Admin SDK 초기화
    cred = credentials.Certificate('Yout key Source') #비공개키 주소를 여기
    firebase_admin.initialize_app(cred)

    registration_token = "Your Fire base tocken"
    message = messaging.Message(
        notification=messaging.Notification(
            title = context,
            body = description
        ),
        token=registration_token
    )

    response = messaging.send(message)
    print('Successfully sent message:', response)



send_message(context, description)

# -----------------------------------

video = None
text = "temp"


##### 요청 오면  MP4 + text 보내기###
app = Flask(__name__)

###-------------------------------
from flask import Flask, jsonify
from PIL import Image
import io
import base64

####
def read_mp4_to_numpy(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Convert to NumPy array: (T, H, W, C)
    video_np = np.stack(frames, axis=0)

    # Transpose to (T, C, H, W)
    video_np = video_np.transpose(0, 3, 1, 2)

    return video_np
####

video_np = read_mp4_to_numpy("video.mp4")
def numpy_to_base64_list(video_np):
    frames_b64 = []
    for i in range(video_np.shape[0]):
        # (3, H, W) -> (H, W, 3)
        frame = video_np[i].transpose(1, 2, 0)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        frames_b64.append(b64_str)
    return frames_b64

@app.route('/video', methods=['GET'])
def get_video():
    frames_b64 = numpy_to_base64_list(video_np) ###이 video_np에 T x C x H x W 도영상 np array로
    return jsonify({'frames': frames_b64})
###-------------------------------



###-------------------------------

@app.route('/text', methods=['GET'])

def send_text(): #Text처리
    data = {
        "situation": context,
        "description": description
    }
    return jsonify(data)
###-------------------------------

app.run(host='0.0.0.0', port=5001)

