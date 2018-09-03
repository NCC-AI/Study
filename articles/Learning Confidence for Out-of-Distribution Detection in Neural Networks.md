# Learning Confidence for Out-of-Distribution Detection in Neural Networks
(2018) Terrance DeVries / Graham W. Taylor  
https://arxiv.org/pdf/1802.04865.pdf  

## どんなもの？（アブストと結論とイントロで読んだものをここに書く）
![LC1](https://github.com/NCC-AI/Study/blob/images/Learning%20Confidence/LC1.png)
- 全く意味のないインプットに対して、高い確率を持つ誤ったアウトプットをしてしまうAI Safetyの問題
- 予測が間違っていることを認識できないという課題があり、この課題関連のタスクをout-of-distribution detectionと呼ぶ
- 本論文はインプットがこれまでのデータセットの範囲内にあるかどうかを判断し、予測の確信度を一緒に出力する手法を提案

## 先行研究と比べて何がすごい？（関連研究で読んだものをここに書く）
- 教師データにラベルを追加したり、ネットワークの出力を利用しない
- 実装が簡単かつアウトプットが直感的

## どうやって有効だと検証した?（実験で読んだものをここに書く）
![LC3](https://github.com/NCC-AI/Study/blob/images/Learning%20Confidence/LC3.png)
- AUROC, AUPRで out-of-distribution detectionの精度を測定
- 様々なデータセットとモデルで実験、他の既存手法との比較

## 技術の手法や肝は？（マテリアル&メソッドで読んだものをここに書く）
![LC2](https://github.com/NCC-AI/Study/blob/images/Learning%20Confidence/LC2.png)
- 例えばテストでヒントを求めれば正解率は上がるけど、同時にペナルティをもらうイメージ
- class prediction branchに並行して、confidence estimation branch（0~1のスカラーを出力、確信度高が1）
- 確信度が1に近いなら通常のクラス分けと同じ、確信度が0に近いなら正解ラベルをもとに補正されるが、ペナルティを受ける
- 結果、そのインプットを誤分類しそうかどうかの確信度を予測することが全体のロス削減に繋がる

## 議論はある？（ディスカッションで読んだものをここに書く）
- 分類タスクにしか適用できない

## 次に読むべき論文は？（参考文献で読んだものをここに書く）
- A baseline for detecting misclassified and out-of-distribution examples in neural networks
- Enhancing the reliability of out-of-distribution image detection in neural networks
