# Multimodal Gesture Recognition Using 3-D Convolution and Convolutional LSTM
(2017) GUANGMING ZHU / LIANG ZHANG / PEIYI SHEN / JUAN SONG  
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7880648

## どんなもの?
- 様々なジェスチャー(21種類)の動画をクラス分けする。
- 3DCNNとConvLSTMを組み合わせた手法。奥行きデータも活用することで精度を上げる。

## 先行研究と比べて何がすごい？
- 3DCNNとConvLSTMを組み合わせるところが初めて。
- Spatial Pyramid Poolingというテクニックを使ってる。

## どうやって有効だと検証した?
- 4万件もジェスチャー動画データセットを用意した。
- 21種類のクラスの動画を5割の精度で認識できるようになった。
- 手法の比較で、他手法よりも2割くらい精度を上げている。

## 技術の手法や肝は？
![conv3d-convlstm](https://github.com/NCC-AI/Study/blob/images/Conv3D_ConvLSTM/Network.png)
- 3-D Convolution + Convolutional LSTMの組み合わせ。
    - 特徴抽出は Conv3Dが優れている。 
    - 長期の関係を抽出するのは LSTMが優れている。
- Late multimodal fusion
RGBの動画に加えて奥行き(Depth)データを利用する時に、入力する時に組み合わせるか、ネットワークの途中で組みわせるかの選択肢がある。
後者の方がどんな形式のデータでもマージしやすいことから後者を選んだ。

- Spatial Pyramid Pooling
全結合層をしてると訓練パラメータが大変なことになるので、減らす工夫
中間層の特徴マップが28×28=784
 => 49 + 16 + 4 + 1 = 70　に減らす
 ![spatial pooling](https://github.com/NCC-AI/Study/blob/images/Conv3D_ConvLSTM/Spatial_Pooling.png)

## 議論はある？
結局、動画のクラス分けのタスクにおいてグローバルに良い手法がどれなのか分からない。
データセットに応じて、いろんな手法を試して精度を確認するしかない。

## 次に読むべき論文は？
ConvLSTMだけの手法とかConv3Dだけの手法とか  
[Learning to Detect Violent Videos using Convolutional Long Short-Term Memory](https://arxiv.org/abs/1709.06531)

*[著者実装](https://github.com/GuangmingZhu/Conv3D_CLSTM)をKerasで実装したい。
