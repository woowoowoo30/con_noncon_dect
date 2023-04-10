import logging as rel_log
import os
import shutil
from datetime import timedelta, datetime
from flask import *
from processor.AIDetector_pytorch import Detector
import csv
import cv2
import time

import core.main

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# 添加 header 解決跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))

# 左右即時影片輸出
@app.route('/video/<path:video>')
def get_video(video):
    path = f'video/{video}/video.mp4'
    mimetype = 'video/mp4'
    with open(path, 'rb') as f:
        video_data = f.read()
    return Response(video_data, mimetype=mimetype)

def generate():
    # 設定截圖間隔（單位：秒）
    interval = 1
    # 計時器
    prev_time = time.time()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 檢查是否已經過了指定的間隔時間
        curr_time = time.time()
        elapsed_time = curr_time - prev_time

        if elapsed_time >= interval:
            # 儲存當前的影像
            cv2.imwrite(f"./tmp/origin/7_11/left/{int(curr_time % 60)}.jpg", frame)
            # 進行預測
            video_y, video_info = current_app.model.detect(frame)
            # 儲存預測的影像
            cv2.imwrite(f'./tmp/draw/7_11/left/{int(curr_time % 60)}.jpg', video_y)
            # 儲存預測的結果
            with open(f'./tmp/txt/7_11/left/{int(curr_time % 60)}.txt', 'w') as f:
                for line in video_info:
                    f.write(f'{line}\n')
            # 更新計時器
            prev_time = curr_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

# 即時影像輸出，每秒儲存，週期60秒
@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 總覽
@app.route('/all')
@app.route('/all/<class_name>')
def all(class_name=None):
    if class_name:
        con_num = 0
        non_con_num = 0
        # 指定目標資料夾路徑
        folder_path = f'./tmp/txt/{class_name}'
        # 使用 os 模組列出所有檔案和資料夾
        for folder in os.listdir(folder_path):
            # 指定目標資料夾路徑
            direction_folder_path = f'{folder_path}/{folder}'
            # 使用 os 模組列出所有的 .txt 檔案
            txt_files = [f for f in os.listdir(direction_folder_path) if f.endswith('.txt')]
            # 逐一讀取每個 .txt 檔案
            for file_name in txt_files:
                file_path = os.path.join(direction_folder_path, file_name)
                with open(file_path, 'r') as f:
                    # 逐行讀取檔案內容
                    for line in f:
                        line_strip = line.strip()  # 去除每行的換行符號
                        line_split = line_strip.split(' ')
                        if line_split[0] == "0":
                            con_num += 1
                        else:
                            non_con_num += 1
        # 計算專心率
        return jsonify({ "rate": round(con_num/(con_num+non_con_num),4) })
    else:
        cls_rate = {}
        # 指定目標資料夾路徑
        folder_path = f'./tmp/txt'
        # 使用 os 模組列出所有班級資料夾
        for cls_folder in os.listdir(folder_path):
            con_num = 0
            non_con_num = 0
            # 班級目標資料夾路徑
            cls_folder_path = f'{folder_path}/{cls_folder}'
            # 使用 os 模組列出所有方向資料夾
            for direction_folder in os.listdir(cls_folder_path):
                # 班級目標資料夾路徑
                direction_folder_path = f'{cls_folder_path}/{direction_folder}'
                # 使用 os 模組列出所有的 .txt 檔案
                txt_files = [f for f in os.listdir(direction_folder_path) if f.endswith('.txt')]
                # 逐一讀取每個 .txt 檔案
                for file_name in txt_files:
                    file_path = os.path.join(direction_folder_path, file_name)
                    with open(file_path, 'r') as f:
                        # 逐行讀取檔案內容
                        for line in f:
                            line_strip = line.strip()  # 去除每行的換行符號
                            line_split = line_strip.split(' ')
                            if line_split[0] == "0":
                                con_num += 1
                            else:
                                non_con_num += 1
            cls_rate[f"{cls_folder}"] = round(con_num/(con_num+non_con_num),4)
        return jsonify(cls_rate)

@app.route("/retrain", methods=['POST'])
def retrain():
    body = json.loads(request.data)
    datas = body["data"]
    for data in datas:
        # 把檔案複製到再次訓練區
        split_url = data["url"].split('/')
        file_name = f"{split_url[-1]}"
        base, ext = file_name.split('.')
        ori_path = f'tmp/origin/{data["url"]}'
        new_path = f'tmp/retrain/{file_name}'
        shutil.copy2(f'{ori_path}', f'{new_path}')
        # 儲存標註檔
        with open(f'{new_path}/{base}.txt', 'w') as f:
            for line in data["info"]:
                f.write(f'{line}\n')
    return jsonify({ "success": 1 })

if __name__ == '__main__':
    files = [
        'uploads', 'tmp/ct', 'tmp/draw',
        'tmp/image', 'tmp/mask', 'tmp/uploads'
    ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    with app.app_context():
        current_app.model = Detector()
    # app.run(host='127.0.0.1', port=5003, debug=True)
    app.run(host='localhost', port=5003, debug=True)
