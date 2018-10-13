# Recurrent Convolutional Neural Network Regression for Continuous Pain Intensity Estimation in Video
(2016) Jing Zhou / Xiaopeng Hong  
https://arxiv.org/pdf/1605.00894.pdf  

## どんなもの?（アブストと結論とイントロ）
![crnn1](https://github.com/NCC-AI/Study/blob/images/CRNN/crnn1.png)
- 新生児や集中治療室の患者は、言語コミュニケーションができないので、表情から痛みの強さを測定することが求められる。
- ConvolutionalRecurrentNNベースの、表情から痛みの強さを予測するリアルタイム回帰モデルを開発した。

## 先行研究と比べて何がすごい？（関連研究）
- 先行研究は静止画に対して解析を行なっている。（本論文は動画）
- 先行研究は分類タスクとしてこういった問題を扱っている。（本論文は回帰タスク）

## どうやって有効だと検証した?（実験）
- UNBC-McMaster Shoulder Pain Expression Archive Databaseを使って評価

## 技術の手法や肝は？（マテリアル&メソッド）
![crnn2](https://github.com/NCC-AI/Study/blob/images/CRNN/crnn2.png)
- Active Appearance Modelで顔の輪郭より内側だけを取り出す
- Recurrentレイヤーの固定サイズのインプットを得るために、スライディングウィンドウを使っている
- 最終FC層では、連続値を出力できるよう、損失関数をMSE、活性化関数をLinearにする

## 議論はある？（ディスカッション）
- trainingにかかる時間に進展の余地があるらしい

## 次に読むべき論文は？（参考文献）
- 探してます
