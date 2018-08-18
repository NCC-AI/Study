# coding:utf-8

import numpy as np
import cv2
from keras import backend as K
from keras.applications.xception import Xception
from keras.preprocessing.image import img_to_array, load_img

K.set_learning_phase(0)  # set learning phase


def grad_cam(model, x, layer_name):
    """
    Args:
       model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    """

    # 前処理
    X = np.expand_dims(x, axis=0)
    X = X.astype('float32')
    preprocessed_input = X / 255.0

    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    #  勾配を取得

    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成

    cam = cv2.resize(cam, (299, 299), cv2.INTER_LINEAR)  # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam


if __name__ == '__main__':
    input_model = Xception(input_shape=(299, 299, 3))
    input_array = img_to_array(load_img('fruits_bowl.jpg', target_size=(299, 299)))
    output_mask = grad_cam(input_model, input_array, 'block14_sepconv2')
    cv2.imwrite('cam_result.jpg', output_mask[:, :, ::-1])

