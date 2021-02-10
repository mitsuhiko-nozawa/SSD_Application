import numpy as np
import matplotlib.pyplot as plt

def show(image, detections, eval_categories, data_confidence_level=0.5):
    predict_bbox, pre_dict_label_index, scores = ssd_predict(image, detections, data_confidence_level)
    vis_bbox(image, bbox=predict_bbox, label_index=pre_dict_label_index, \
        scores=scores, label_names=eval_categories)

def ssd_predict(image, detections, data_confidence_level):
    # confidence_levelが基準以上を取り出す
    height, width, channels = image.shape
    predict_bbox = []
    pre_dict_label_index = []
    scores = []
    detections = detections[0]

    # 条件以上の値を抽出
    #find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
    find_index = np.where(detections[0:, :, 0] >= data_confidence_level)
    detections = detections[find_index]
    for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
        #print(find_index[0][i])
        #if (find_index[1][i]) > 0:  # 背景クラスでないもの
        if (find_index[0][i]) > 0:  # 背景クラスでないもの
            sc = detections[i][0]  # 確信度
            print(sc)
            bbox = detections[i][1:] * [width, height, width, height]
            # find_indexはミニバッチ数、クラス、topのtuple
            lable_ind = find_index[0][i]-1
            # lable_ind = find_index[1][i]-1
            # （注釈）
            # 背景クラスが0なので1を引く

            # 返り値のリストに追加
            predict_bbox.append(bbox)
            pre_dict_label_index.append(lable_ind)
            scores.append(sc)

    return predict_bbox, pre_dict_label_index, scores

def vis_bbox(rgb_img, bbox, label_index, scores, label_names):
    # 枠の色の設定
    num_classes = len(label_names)  # クラス数（背景のぞく）
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    # 画像の表示
    #fig = plt.figure(figsize=(10, 10))
    fig = plt.figure()
    plt.imshow(rgb_img)
    currentAxis = plt.gca()

    # BBox分のループ
    for i, bb in enumerate(bbox):
        # ラベル名
        label_name = label_names[label_index[i]]
        color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

        # 枠につけるラベル　例：person;0.72　
        if scores is not None:
            sc = scores[i]
            display_txt = '%s: %.2f' % (label_name, sc)
        else:
            display_txt = '%s: ans' % (label_name)

        # 枠の座標
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]

        # 長方形を描画する
        currentAxis.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=2))

        # 長方形の枠の左上にラベルを描画する
        currentAxis.text(xy[0], xy[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    fig.savefig("img.png")