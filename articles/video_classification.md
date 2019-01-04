# タイトルをここに書く
(2015) Joe Yue-Hei Ng/Matthew Hausknecht/Sudheendra Vijayanarasimhan/Oriol Vinyals/Rajat Monga/George Toderici
https://arxiv.org/pdf/1503.08909v2.pdf

## どんなもの?
動画のクラス分類について、実験的に精度比較
![video_classification](https://github.com/NCC-AI/Study/blob/images/video_classification/summary.png)

## 先行研究と比べて何がすごい？
いろんな手法を比較して、どれが良いか教えてくれたところ。

## どうやって有効だと検証した?
動画データセット(UCF-101 datasets)

## 技術の手法や肝は？
Convlution + LSTM　と、 Convlution + 時系列に展開してPoolingの比較
- 結局LSTMは精度良くない。
- Conv + Poolingが良い。
- Optical Flowは、ほんのちょっぴり精度向上。
![conv pooling](https://github.com/NCC-AI/Study/blob/images/video_classification/conv+pooling.png)
![lstm](https://github.com/NCC-AI/Study/blob/images/video_classification/lstm.png)

## 議論はある？
- LSTMは、動画の時系列は良くないということが曖昧に感じてたけど、やはりか感
    - 位置情報の時系列的相関が失われるのは痛い。
- BiLSTMにしたら精度が上がるけど、それでもConv Poolingに勝てないかな？
- Optical Flowは元画像も同時に入力しないと情報が落ちてしまって精度出ないので気をつける。

## 次に読むべき論文は？
2015年以降で、どのような研究が行われモデルが改良されてきたか知りたい。
