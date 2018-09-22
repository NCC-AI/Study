# Automated Classification of Lung Cancer Types from Cytological Images Using Deep Convolutional Neural Networks
(2017) Atsushi Teramoto / Tetsuya Tsukamoto （藤田保健衛生大、岐阜大）
https://www.researchgate.net/publication/319100618_Automated_Classification_of_Lung_Cancer_Types_from_Cytological_Images_Using_Deep_Convolutional_Neural_Networks

![2018-09-23 0 43 13](https://user-images.githubusercontent.com/12442472/45918969-b178fb80-bec9-11e8-9046-d8779f57f5a2.png)

## どんなもの?
肺がんのタイプを診断する病理医のタスクを自動化する。<br>
量が少ないオリジナルのデータを使って、Data Augmentationを施し学習させたところ人間に匹敵する71.1%のAccuracyが得られた。

## 先行研究と比べて何がすごい？
がんの種別の診断はその後の治療方針を決定するために重要であるが、病理医が不足している。<br>
病理診断医おいては組織診断も重要であるがその形態からの判断も限界があり、肺がんにおいては細胞画像からの診断が有効である。<br>
がんの種別の自動診断において組織病理に対してDeep Learningを適用している先行研究はあるが、細胞画像に対して適用した例はない。<br>
学習結果において成績が向上した理由や、うまくいかなかった例についても考察している。


## どうやって有効だと検証した?
![2018-09-23 0 41 36](https://user-images.githubusercontent.com/12442472/45918959-95755a00-bec9-11e8-81a1-d5d254ebe65b.png)
![2018-09-23 0 53 57](https://user-images.githubusercontent.com/12442472/45919072-2e58a500-becb-11e8-80d1-233a69523077.png)<br>
弁別の対象はAdenocarcinoma, Squamous cell carcinoma, Small cell carcinomaの3つ。<br>
自作のCNNで60000 epoch（8h）学習させ、3-Fold Cross Validationを行い性能を評価した。<br>


## 技術の手法や肝は？
それぞれ症例数は40, 20, 16<br>
スライドガラス全体の組織から768x768（リサイズ後は256x256）の領域をcropし、それぞれ82, 125, 91枚取得。<br>
Data Augumentationは実際に有りうる画像のバリエーションの範囲内として、rotation, invertation, filtering（ Gaussian, Edge Filter ）
で行った。CrossーValidation時の各ユニットが5000程度になるように拡張した。（Filteringは顕微鏡のボケというローカルな制約を考慮したため入っている。）<br>
各クラスごとにどのようにうまくいったのか判断するためConfusion Matrix（混同行列）で議論を行った。


## 議論はある？
1. 3つの種類の弁別では71.1%のAccuracyだった。人間と同程度である。
![2018-09-23 1 09 43](https://user-images.githubusercontent.com/12442472/45919261-78428a80-becd-11e8-92c1-8f0af636ad0e.png)
2. もっとも重要なsmall-cell carcinomaか否かのAccuraryでは85.6%であり十分と考えられる。
3. Squamous cell carcinomaの成績が低いことの原因としては、①Adenocarcinomaと見た目が似ている ②元の症例数が少ないため般化が難しかった の2つが考えられる。
4. 症例数の少ないsmall cell carcinomaは他の例と形態が特徴的に異なるため十分な弁別が可能な特徴量が設計できたと考えられる
![2018-09-23 1 09 38](https://user-images.githubusercontent.com/12442472/45919259-75e03080-becd-11e8-9e5d-514b5e4d7bcb.png)
![2018-09-23 1 09 12](https://user-images.githubusercontent.com/12442472/45919253-652fba80-becd-11e8-896c-83f6c9a64058.png)
5. Data Augmentationの効果としては、Adenocarcinoma、Squamous cell carcinomaのAccucracyが15%以上改善された。一方で、small cell carcinomaは5%低下した。



## 次に読むべき論文は？
Introductionのところで気になったものに、手動設計の特徴量とDCNNを組み合わせた手法を使った研究があったので確認したい。<br>

1. H.Wang,A.Cruz-Roa,A.Basavanhallyetal.,“Mitosisdetec-tion in breast cancer pathology images by combining hand-craed and convolutional neural network features,” Journal ofMedical Imaging
https://www.researchgate.net/publication/266734716_Mitosis_detection_in_breast_cancer_pathology_images_by_combining_handcrafted_and_convolutional_neural_network_features

同じようなことやってる論文という意味ではこちら<br>
2. Automatic Classification of Ovarian Cancer Types from Cytological Images Using Deep Convolutional Neural Networks
https://www.researchgate.net/publication/323978845_Automatic_Classification_of_Ovarian_Cancer_Types_from_Cytological_Images_Using_Deep_Convolutional_Neural_Networks



