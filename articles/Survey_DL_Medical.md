# A Survey on Deep Learning in Medical Image Analysis
(2017) Geert Litjens, Thijs Kooi, et al, Radboud University Medical Cente  
https://arxiv.org/pdf/1702.05747.pdf

## Survey論文ということでまとめ方が違います、ご容赦ください

## どんなもの?
Deep Learningの医療応用について300以上の研究をまとめたSurvey論文。
医療応用における特殊性を事例を交えて紹介し、キーとなった発想やノウハウまで説明している。
### イントロ
・DNNの歴史
・DNNの手法
・Classificationアーキテクチャ
・Multi-Streamアーキテクチャ
・Segmentationアーキテクチャ
・RNN
・教師なし深層学習
・RBM

### Medical ImagingにおけるデDeep Learningの使い方について
・Classification
・Detection
・Segmentation
・Registration
・画像からの類似症例検索
・Image generation and enhancement
・画像と説明文作成

### 解剖学的領域ごとの紹介
・脳
・眼
・胸部
・病理
・乳房
・心臓
・腹部
・筋骨格系
・その他

### 領域ごとの論文の要約表

### Discussion


## 先行研究と比べて何がすごい？
タスク別・領域別にまとめているため関係ありそうな研究トピックが見つけやすい

## どうやって有効だと検証した?
とにかく大量の論文を対象にしているのである程度一般的な流れを説明できていると思われる

## 技術の手法や肝は？
### 医療領域への応用の課題の特殊性は？
・学習データが少ない（正確にはラベルづけされたデータが少ない）
・Classificationにおけるクラスのアンバランス

### 医療データは基本的に数が少ないので事前学習や転移学習を行うべし
・事前学習はAutoEncoderとか
・転移学習はVGGなど大量の自然画像でトレーニングされたものの特徴量抽出部分を使うと良い
・特徴量抽出部分を固定するかそこもトレーニング対象にするかは明確にどちらが良いかは決まっていない

### Multi-Stream アーキテクチャ
・異なるデータ・タスクの結果を統合するような場合適している
・Contextが重要なタスクではマルチストリームでマルチスケールの特徴抽出が有効
・３DCNNでもいいがパラメータが多くなる

### Segmentation アーキテクチャ
・Sliding-windowではwindowごとに同じ計算するので時間かかる
・FCNNは解像度が問題
・U-Net構造なら解像度問題解決

### RNN
・時系列解析に有利
・LSTM

### 教師なし深層学習
・AutoEncoderで事前学習おすすめ
・RBM

### Classificationタスク
・データ少なくなりがちなので転移学習を利用しよう
・転移学習の効果を検討した論文の説明
・昔はSAEやRBM使ってたがいまは専らCNN
・MRIからのアルツハイマー判定とか

### Detectionタスク
・ランドマークを配置する例
・3Dデータをよく使うのでRNN,LSTM, 3DCNN
・MultiーStream CNNでCTとPETの情報を統合したり
・クラスのアンバランスに不正解データを多く学習させて対応した例
・MRIから心臓の拡張収縮を検出とか

### Segmentationタスク
・医療では体積・形状をパラメータ化したいことが多い
・U-Netを3D対応にしたV=net
・RNNでセグメンテーションも一般的
・U-NetとLSTMの混成も
・クラスのアンバランスへの対応で、損失関数に特異度と感度を適当に重み付けする
・クラスのアンバランスへの対応で、陽性データ（少ない）のみデータ拡張を行う

### Registration
・CTとMRIの位置合わせしたり
・①画像の類似度を反復的に見て調整する方法
・②変換パラメータを回帰ネットワークで学習するアプローチ
・まだ確立はしていない

### 画像検索
・類似症例を簡単に検索したい
・医学的に意味のある類似をみつける
・まだ確立していない

### Image Generation and Enhancement
・MRI画像からPET画像作ったり
・X線で骨の像消したり
・DLならではなので楽しみ

### 画像と説明文作成
・画像から説明文生成
・画像だけでなく文章からも判定を行ったり



## 議論はある？
・とにかく発展が早い
・重要なのはアーキテクチャだけでなくむしろ前処理や適切なデータ拡張だという人も多い
・最適なハイパーパラメータを明確な方法はない
・画像に付随する文書を利用する場合、PACSであれば大量に存在するがテキストマイニング技術が必要になる
・画像だけでなく統計データも利用すると良いと考えられるがいまのところ大幅な改善に至った例はない
・VAEやGANもこれから楽しみ
・中間層が何をしているか確かめるための手法もいくつか提案されている

## 次に読むべき論文は？
まずは自分の取り組むタスクについてこの資料で調べて該当する論文を読むのが良いかと思う

自分として興味があるのは
Zeiler, M. D., Fergus, R., 2014. Visualizing and understanding convolutional
networks. In: European Conference on Computer Vision.
pp. 818–833.

Springenberg, J. T., Dosovitskiy, A., Brox, T., Riedmiller, M., 2014.
Striving for simplicity: The all convolutional net. arXiv preprint
arXiv:1412.6806.

Montavon, G., Lapuschkin, S., Binder, A., Samek, W., Muller, K.- ¨
R., 2017. Explaining nonlinear classification decisions with deep
taylor decomposition. Pattern Recognition 65, 211–222
