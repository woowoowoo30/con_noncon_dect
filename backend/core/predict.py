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
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
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

def predict_appear(data, target):
    # 讀取數據集
    df = pd.read_csv('./tmp/appeardatas/許厝港鳥類調查記錄資料處理.csv')

    # 拆分成特徵欄位和目標變量
    df_features = df.iloc[:, :8] 
    df_target = df.iloc[:, target:target+1]

    df_target_rep = np.array(df_target)
    df_target_rep = np.where(df_target_rep > 0, 1, 0)
    df_target_rep = pd.DataFrame(df_target_rep)

    # print(df_features)
    # print(df_target)

    # 切割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target_rep, test_size=0.2)

    # 使用SVM進行二元分類
    clf = SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(X_train, y_train)

    # 在測試集上測試準確度
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    dateTime = data["date"]
    dateTime = datetime.strptime(dateTime, "%Y-%m-%d %H:%M")
    date = dateTime.strftime("%-m%d")
    time = dateTime.strftime("%-H%M")
    weather = data["weather"]
    temperature = data["temperature"]
    windDirection = data["windDirection"]
    windForce = data["windForce"]
    humidity = data["humidity"]
    tidal = data["tidal"]

    # 預測鳥類是否會出現
    new_data = pd.DataFrame({
        '日期': date,
        '時間': time,
        '天氣': weather, 
        '氣溫': temperature, 
        '風向': windDirection, 
        '風力': windForce, 
        '相對濕度': humidity, 
        '潮汐': tidal
    }, index=[0])
    y_pred = clf.predict(new_data)
    y_pred_cna = '會出現' if y_pred == 1 else '不會出現'
    y_pred_num = 0

    if y_pred == 1:
        df2 = df
        df2 = df2.loc[(df2.iloc[:, target:target+1]!=0).any(1)]
        df_features2 = df2.iloc[:, :8] 
        df_target2 = df2.iloc[:, target:target+1]
        # 劃分訓練及測試集
        X_train, X_test, y_train, y_test = train_test_split(df_features2, df_target2, test_size=0.2)
        # 訓練模型
        regressor = DecisionTreeRegressor(random_state=42)
        regressor.fit(X_train, y_train)
        # 預測結果
        y_pred = regressor.predict(X_test)
        # 將預測結果轉換為 DataFrame
        df2_pred = pd.DataFrame(y_pred, columns=df_target.columns)

        y_pred_num = regressor.predict(new_data)
        df2_pred_num = pd.DataFrame(y_pred_num, columns=df_target.columns)

        y_pred_num = df2_pred_num.iloc[0][0]
    
    return { 'accuracy': round(accuracy, 2), 'result': y_pred_cna, 'number': round(y_pred_num) }