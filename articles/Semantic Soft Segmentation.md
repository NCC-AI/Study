# Semantic Soft Segmentation
(2018) YAĞIZ AKSOY / TAE-HYUN OH    
http://cfg.mit.edu/sites/cfg.mit.edu/files/sss_3.pdf

## どんなもの?
- 物体を正確に形取ったレイヤーを作成し、マスクや合成など簡単に画像の編集ができるメソッド
- 物体検出やクラス分類が目的ではなく、高精度のセグメンテーションの達成を目指した

![SSS_1](https://github.com/NCC-AI/Study/blob/images/Semantic%20Soft%20Segmentation/SSS_1.png)

## 先行研究と比べて何がすごい？
- 髪や草などの細かい部分まで含めて、正確に形取る
- 途中操作や専門知識を必要としない完全な自動化
- アーティストが他の編集ツールでも使用できる画像表現

## どうやって有効だと検証した?
- 関連方法と定性的に比較し、特徴的な違いについて議論

![SSS_2](https://github.com/NCC-AI/Study/blob/images/Semantic%20Soft%20Segmentation/SSS_2.png)

## 技術の手法や肝は？
- Input imageからの流れ
  - Generated semantic features, ニューラルネットワークが、画像の領域と特徴を推測する
  - Matting affinity, 髪や草などの変わり目を検出する（従来は人の手によって分けられねばならなかった）
  - Color affinity, 画像全体のピクセルは色によって相互に関連づけられる
  - Semantic affinity, ニューラルネットワークが検出した特徴と結合される
- ラプラシアン行列がピクセルのペアが同じセグメントに属する可能性を表すことがコアらしい

![SSS_3](https://github.com/NCC-AI/Study/blob/images/Semantic%20Soft%20Segmentation/SSS_3.png)

## 議論はある？
- 正確だが、スピードに課題（640×480の画像に3,4分かかる）
- 1つのオブジェクトが複数のレイヤーに分割されることがある
- 同じクラスの複数のインスタンスを別々のレイヤーで表現することができない
- 大きい範囲で物体同士の色が似ていたり、信頼性の低いセマンティック特徴ベクトルが生成されると失敗する

## 次に読むべき論文は？
- セマンティックセグメンテーションに関する論文
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)
