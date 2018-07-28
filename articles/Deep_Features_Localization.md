# Learning Deep Features for Discriminative Localization
(2016) Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba  
Computer Science and Artificial Intelligence Laboratory, MIT  
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

## どんなもの?
Deep Learning がクラス分けタスクにおいて判断の理由が分からない問題の解決に取り組んだ

## どうやって有効だと検証した?
判断の理由を可視化した結果を見て、人間の感覚と合ってそうだと思った。
同じ画像でも、クラス分けの結果の違いによって、可視化した結果も異なっている。

## 技術の手法や肝は？
- CNNのネットワークの最終層で特徴マップを入力画像サイズに合わせてGlobal Average Poolingでクラス分けを行う。
- (特徴マップ)*(最終層の結合の重み)を利用して、どの特徴マップが答えに影響していたかの重み付け画像が作れる。

## 議論はある？
可視化のためのネットワーク構造にすることで、学習精度が下がることはないのか？

## 先行研究と比べて何がすごい？
Global Average Poolingでクラス分けの判断の理由を可視化したのは初めて

## 次に読むべき論文は？
Class Activation Mapping(CAM)と呼ばれて有名な技術の一つになった。
CAMの派生形が、最近増えている。より局所的になったり、学習精度が落ちないような工夫を次に調査する
