# Learning to Segment Every Thing
(28 Nov 2017) Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, Ross Girshick

https://arxiv.org/pdf/1711.10370.pdf

## どんなもの?
- セグメンテーションタスクは学習データ全てをセグメンテーションマスクする必要があって大変
- detectionの重みをsegmentationの重みへ転移学習

## 先行研究と比べて何がすごい？
- Mask R-CNNというインスタンスセグメンテーションする手法があるが，全ての訓練データにマスクアノテーションが必要
- NLPなどで予測するのはあった
- detectionの重みを直接学習に用いる

## どうやって有効だと検証した?
実験で読んだものをここに書く
- COCO2017を使って学習
- 

## 技術の手法や肝は？
(Figure 2)
- ResNet-50-FPN / ResNet-101-FPN ベース(Imagenetでpretrain)
- detectionネットワークとmaskネットワークを途中でつなげてパラメータを転移
- 誤差逆伝播の際にmaskネットワークからdetectionネットワークへエラーが伝わってしまうため，重みの転移はdetecton→maskの一方通行にした


## 議論はある？
ディスカッションで読んだものをここに書く


## 次に読むべき論文は？
- J. Pennington, R. Socher, and C. Manning. GloVe: Global
vectors for word representation. In EMNLP, 2014. 2, 3, 5
本論文と比較されてたNLPを使った手法
