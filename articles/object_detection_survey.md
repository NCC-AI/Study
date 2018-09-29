# Object Detection Survey
(2018.9) Li Liu / Wanli Ouyang  
https://arxiv.org/pdf/1809.02165.pdf

# （前編）（長すぎる）

## survey論文のため構成を少し変更します、ご容赦ください。

## どんなもの?
・Deep Learning登場以降で一般物体検出のSurvey論文がなかったので作った。<br>
・トップカンファの論文のみ250報から作成<br>
・今までの一般物体検出の歴史から追ってこの先どうなるかを議論<br>
・30Pある。わりとつらい。<br>


## 構成
1. Introduction<br>
なぜこの論文を作ったのかの説明
2. Background<br>
2.1 The Problem<br>
・何が原因で検出タスクが難しいか<br>
2.2.1 Accuracy Related Challenges<br>
・一般物体検出におけるAccuracyの向上を阻む要因は何か<br>
2.2.2 Efficiency Related Challenges<br>
・なぜEfficiencyが問題になるか<br>
2.3 Progress in the Past Two Decades<br>
・この20年間でどのようなチャレンジが行われてきたか（SIFTなどDeep Learning以外を含む）、これからのチャレンジは何か<br>
3. Frameworks<br>
・手法の説明<br>
3.1 Region Based (Two Stage Framework)<br>
・Deep Learningを使ったtwo-stage手法の歴史と各手法の発想・特徴、（RCNN, SPPNet, FastRCNN, FasterRCNN, RFCN, Mask RCNN, Light Head RCNN）<br>
3.2 Unified Pipeline(One Stage Pipeline)<br>
・Deep Learningを使ったone-stage手法の歴史と各手法の発想・特徴<br>（Detector Net, Overfeat, YOLO, YOLOv2, YOLO9000, SSD）
4. Fundamental SubProblems
・一般物体検出の基本的な問題
5. Datasets and Performance Evaluation
・人気のデータベースと最新の性能
6. まとめ
・今後の方向性

## 内容ダイジェスト
1. Introduction<br>
・DLの一般物体検出の論文まとめたかった<br>
・Specificな（車の検出とか）はいれない<br>
・少なくとも20, 多くて200とか。でも最近はもっと多い<br>
<br>

2. Background<br>
2.1 The Problem<br>
・Structualな対象に絞る（人間は空とか認識するけどそういうのはやらない）<br>
・なぜ難しいのか<br>
![image](https://user-images.githubusercontent.com/12442472/46250807-d64b1100-c47d-11e8-850a-e6c60d5a969a.png)<br>
・やり方としてBoundingBoxが一番流行ってる。Segmentationはその先、難しい<br>
スクリーンショット 2018-09-30 6.50.30<br>
・幾何的情報→手動特徴量→そのカスケード→SIFT→DNN<br>

2.2.1 Accuracy Related Challenges<br>
・なぜ難しいか？<br>
![image](https://user-images.githubusercontent.com/12442472/46250828-33df5d80-c47e-11e8-924d-aff60f352916.png)<br>
![image](https://user-images.githubusercontent.com/12442472/46250830-393ca800-c47e-11e8-9c83-0a8899c8f65d.png)<br>
・classが多いと大変<br>
・人間並みにするなら10^4~10^5オーダー必要<br>
2.2.2 Efficiency Related Challenges<br>
・なぜEfficiencyが問題になるか<br>
・mobileデバイスの需要が高まっている<br>
・簡単でないとクラスが増えた時しんどい、そういう時はweak superviser learningを使うべき
2.3 Progress in the Past Two Decades<br>
・この20年間でどのようなチャレンジが行われてきたか（SIFTなどDeep Learning以外を含む）、これからのチャレンジは何か<br>
3. Frameworks<br>
・手法の説明<br>
3.1 Region Based (Two Stage Framework)<br>
![image](https://user-images.githubusercontent.com/12442472/46250773-e2829e80-c47c-11e8-8b5c-698f6dfd763b.png)
<br>・Deep Learningを使ったtwo-stage手法の歴史と各手法の発想・特徴、（RCNN, SPPNet, FastRCNN, FasterRCNN, RFCN, Mask RCNN, Light Head RCNN）<br>


・RCNN<br>
・複雑すぎる。<br>
・工程が分かれすぎ<br>
・領域判定が荒すぎる<br>
・SVMのせいで容量とパワーやばい<br>
・複雑すぎる。<br>

![image](https://user-images.githubusercontent.com/12442472/46250834-57a2a380-c47e-11e8-8179-9f335cd7643d.png)<br>
・SPPNet<br>
従来からあったSPPを使ってCNNと接続。でもBackPropできなくなるのでFine-Tuningができない。困った。<br>
![image](https://user-images.githubusercontent.com/12442472/46250835-5bcec100-c47e-11e8-9400-db5888155311.png)<br>
![image](https://user-images.githubusercontent.com/12442472/46250836-5ec9b180-c47e-11e8-9e0d-9c22bfbacc23.png)<br>
![image](https://user-images.githubusercontent.com/12442472/46250839-61c4a200-c47e-11e8-8e65-447d2d9fea7f.png)<br>
・FastRCNN<br>
resionはちがうけどそのあとはend-to-end<br>
・FasterRCNN<br>
もはやその部分がボトルネックだったのでresion proposalを一緒にした<br>
5fps<br>
RPN<br>
・RFCN<br>
わからん（来週）<br>
・Mask RCNN<br>
RoI Poolingの弱点を克服<br>
・Light Head RCNN<br>
わからん（来週）<br>

3.2 Unified Pipeline(One Stage Pipeline)<br>
・Deep Learningを使ったone-stage手法の歴史と各手法の発想・特徴<br>（Detector Net, Overfeat, YOLO, YOLOv2, YOLO9000, SSD）
・Light Head RCNN<br>
わからん（来週）<br>

4. Fundamental SubProblems
・一般物体検出の基本的な問題
5. Datasets and Performance Evaluation
・人気のデータベースと最新の性能
6. まとめ
・今後の方向性

