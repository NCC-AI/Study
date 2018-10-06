# Object Detection Survey
(2018.9) Li Liu / Wanli Ouyang  
https://arxiv.org/pdf/1809.02165.pdf

# （後編）

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

4. Fundamental SubProblems<br>
・一般物体検出の基本的な問題<br>
4.1 DCNN based Object Representation<br>
・物体検出においても良い特徴表現が一番大事、昔はHOGやSIFTなど局所特徴量を抽出する方法を設計したりしていたが、これは深い専門知識を必要とした。<br>
・深層学習はこれを、データのみから抽象的な特徴量を複数得ることができる。これに伴って、必要とされる知識は専門知識からネットワークの設計へと移った。<br>
・Objectのスケール、ポーズ、視点、変形、その他幾何学的変化に頑健なことが求められる。<br>
4.1.1 Popular CNN Architectures<br>
・CNNのアーキテクチャの進化の流れ<br>
・AlexNet、ZFNet、VGGNet、GoogLeNet、Inception<br>
![image](https://user-images.githubusercontent.com/12442472/46574378-08f48c80-c9dd-11e8-9210-7d70da474937.png)<br>
・表より、アーキテクチャの進化の方向は①層が深くなること ②FC層の使用を避けること又はInceptionモジュールの使用によってパラメータ数を削減すること 
③ ResNetなどのショートカット接続によって学習効率をあげること<br>
・その他、DenseNets、Squeeze and Excitation Block, Dilated Residual Networks, Xception, DetNets, Dual Path Networksなどの
基本ネットワークが出現している。<br>
・位置情報を含むデータセットで事前学習することで精度が上がることが示されている。当然ながらこの場合事前学習のデータセットと適用データとの間に類似性があると良い。<br>
・古典的な手法では、広い範囲でオブジェクトを正確に検出する方法は、メモリ容量と処理時間に課題がある。（RCNNとか？）<br>
・CNNを複数回かけて検出を行う場合、CNNの浅い側では空間分解能はあるが抽象的な情報が少ない。深い側では意味論的な情報は取得できる（ポーズ、変形など）
が空間分解能が小さいため幾何学的な情報が失われる。<br>
・上記の問題を解決するために、複数の段階のCNN層の情報から位置と意味論的推定を行う手法が研究されている。<br>
・複数の深さにあるCNNの情報の統合方法として、U-Netのようなスキップ結合は、次元が高すぎて良い結果を出さない。<br>
・その中でもベストな組み合わせとして代表的な方法は、SharpMask, DSSD, FPN, TDM, RON, ZIP, STDN, RefineDet, StairNet<br>
![image](https://user-images.githubusercontent.com/12442472/46574691-3db71280-c9e2-11e8-8d2a-4358d1b2b311.png)<br>
・これらの方法は下図のように、通常のボトムアップ構造に加えて横方向やトップダウン構造の接続を行っている。
![image](https://user-images.githubusercontent.com/12442472/46574715-c6ce4980-c9e2-11e8-96c0-5dfc2fad3dec.png)<br>
・その他、モデルの変形や移動を明示的に表現してから深層学習に与えるためのDeformable Part based Models (DPMs)（Histogram of Gardientの進化版っぽい）<br>
が提案されている。（自分としては抽象表現を獲得する前に人間が明らかに設定できる事前条件を組み込むのは賛成）<br>

4.2 Context Modeling<br>
・画像からの情報だけでなく、事前の条件としてコンテキスト（文脈）を利用したい。現在の手法はコンテキストを考慮できていない。<br>
・意味論的コンテキスト：この場面にはこういうものが ある/ない<br>
・空間的コンテキスト：このオブジェクトのこの位置にはこういうものが ある/ない<br>
・大きさコンテキスト：このオブジェクトがこのサイズならこのオブジェクトはこのサイズの範囲内である<br>





5. Datasets and Performance Evaluation
・人気のデータベースと最新の性能
6. まとめ
・今後の方向性
