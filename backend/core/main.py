from core import process, predict

def c_main(path, model, ext):
    image_data = process.pre_process(path)
    image_info = predict.predict(image_data, model, ext)

    return image_data[1] + '.' + ext, image_info

def d_main(data):
    res = {}
    bird_dict = {
        8: "小燕鷗",
        9: "太平洋金斑鴴",
        10: "東方環頸鴴",
        11: "麻雀",
        12: "黑腹濱鷸"
    }
    for i in range(8,13):
        res[bird_dict.get(i)] = predict.predict_appear(data, i)

    return res

if __name__ == '__main__':
    pass
