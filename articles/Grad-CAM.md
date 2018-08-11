# Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization
(2017) Ramprasaath R. Selvaraju1, Michael Cogswell
https://arxiv.org/pdf/1610.02391.pdf

## どんなもの?
画像のクラス分けタスクの理由可視化/CAM(Class Actication Mapping)をさらに局所的に詳細に可視化できるようになった。
CAMのためにネットワークを変えないで済む。  
(アーキテクチャによらず、既存のモデルで判断理由の可視化ができる。)
![grad-cam-output](https://github.com/NCC-AI/Study/blob/images/grad-cam-output.png)

## 先行研究と比べて何がすごい？
従来のCAMは、可視化と学習精度がtrade offになっていたが、今回のGrad-CAMは、
学習時のモデルを自由に設計できる。(classification以外にも,captioning,reinforcement learningなど,なんでも使える)  
Grad-CAM用の再学習も不要

## どうやって有効だと検証した?
Image Captioningとか、VQA(Visual Question Answering)のタスクに対しても、
答えに応じて、Grad-CAMで可視化された判断理由が正しく表示された。

## 技術の手法や肝は？
1. まずは単にモデルの学習をする
1. 対象クラスのみ1にその他を0にしてから勾配を逆伝搬させる
1. 特徴マップの各チャンネルで勾配の平均をとり重みとする
1. 特徴マップに重みをかけて、全てを足し合わせる(Grad-CAM)
1. 従来のピクセルレベルの手法と組み合わせる

## 議論はある？
- まだまだ、局所性がセグメンテーションには劣る
- もともとのモデルが深くなれければ、CAMの精度は落ちる

## 次に読むべき論文は？
Grad-CAM++(https://arxiv.org/pdf/1710.11063.pdf)
さらに精度をあげているようだ。
![grad-cam++](https://github.com/NCC-AI/Study/blob/images/grad-cam%2B%2B.png)
