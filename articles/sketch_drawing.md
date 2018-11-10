# A Neural Representation of Sketch Drawings
(2017) David Ha / Douglas Eck (Google Brain)
https://arxiv.org/pdf/1704.03477.pdf

## どんなもの?
VAEとLSTMで、スケッチを自動生成(ペンがどのように動いたか再現)

## 先行研究と比べて何がすごい？
- ベクトルイメージデータセットを7万件用意したこと(オープンソース化した)
-  LSTMを適用しこと

## どうやって有効だと検証した?
訓練: 70000, 検証: 2500, テスト: 2500
![Reconstruction](https://github.com/NCC-AI/Study/blob/images/Sketch-RNN/reconstruction.png)

## 技術の手法や肝は？
![Sketch-RNN](https://github.com/NCC-AI/Study/blob/images/Sketch-RNN/sketch-rnn.png)

S = (Δx,Δy,p1,p2,p3)   
Δx: xの変位  
Δy: yの変位  
p1:スケッチが続く  
p2:紙から離れる  
p3:スケッチ終了  
p1, p2, p3は one-hot  

(Δx,Δy) : Gaussian mixture model
(q1,q2,q3) : カテゴリ分類

## 議論はある？
ベクトルのように足し算、引き算できる。
![vector calculate](https://github.com/NCC-AI/Study/blob/images/Sketch-RNN/vector-calculate.png)

不完全なスケッチから予測
![predict ending](https://github.com/NCC-AI/Study/blob/images/Sketch-RNN/predict_ending.png)

## 次に読むべき論文は？
上手な手術の軌跡がどのようなものか定量化して分類ができるのでは？？

