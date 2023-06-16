import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

def predict(dataset, model, ext):
    global img_y
    # x: 檔案路徑
    x = dataset[0].replace('\\', '/')
    # 檔案名稱
    file_name = dataset[1]
    print(file_name)

    # 預測好之後要回傳的陣列，一張圖片占一個位置
    predict_info = []
    
    if (ext in ['png', 'jpg', 'jpeg']):
        # 圖片
        x = cv2.imread(x)
        img_y, image_info = model.detect(x)
        predict_info.append({ "url": f"http://127.0.0.1:5501/tmp/ct/{file_name}.{ext}", "info": image_info })
        cv2.imwrite(f'./tmp/draw/{file_name}.{ext}', img_y)
    else:
        # 影片
        # 影片做法原理是根據影幀切割成多張圖片，每張辨識完之後再組合回影片

        # 讀取影片，並抓相關參數
        cap = cv2.VideoCapture(x)
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(frame_width, frame_height)

        # 初始化圖片組合影片套件
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        videoWriter = cv2.VideoWriter(f'./tmp/draw/{file_name}.{ext}', fourcc, frame_fps, (frame_width, frame_height))
        # 要組合的所有圖片
        img_array = []

        # 檢查影片是否讀取成功
        if (cap.isOpened() == False): 
            print("影片異常")
        # 自動生成資料夾
        if not os.path.isdir(f'./tmp/ct/{file_name}'):
            os.makedirs(f'./tmp/ct/{file_name}')
        if not os.path.isdir(f'./tmp/draw/{file_name}'):
            os.makedirs(f'./tmp/draw/{file_name}')
        # 遍歷影幀索引列表，存儲對應的圖像
        while(cap.isOpened()):
            ret, frame = cap.read()
            index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if ret:
                # 存儲圖像
                cv2.imwrite(f'./tmp/ct/{file_name}/_{index}.jpg', frame)
            else: 
                break
        # (預測)連續讀取影片直到結束
        # 設定讀取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while(cap.isOpened()):
            # 取得影幀
            ret, frame = cap.read()
            index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if ret == True:
                # 進行預測
                video_y, video_info = model.detect(frame)
                # 存儲圖像
                cv2.imwrite(f'./tmp/draw/{file_name}/_{index}.jpg', video_y)
                # 預測結果加入回傳陣列
                predict_info.append({ "url": f"http://127.0.0.1:5501/tmp/ct/{file_name}/_{index}.jpg", "info": video_info })
                # 影幀加入組合陣列
                img_array.append(video_y)
            else: 
                break
        # 釋放
        cap.release()
        # 關閉所有的圖片
        cv2.destroyAllWindows()

        # 開始將圖片組合影片
        for i in range(len(img_array)):
            img = img_array[i]
            img.dtype = np.uint8
            videoWriter.write(img)
        # 組合完之後釋放
        videoWriter.release()

    print('預測完成～')

    return predict_info