# Improved Techniques for Training GANs
(2016) Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
https://arxiv.org/pdf/1606.03498.pdf

## どんなもの?
注目したのは5章の`Semi-supervised learning`  
クラス分類にGANを入れて、Kクラス分類+1(GANによる生成)を分類するタスクにすることで学習精度を向上させる  
MNISTを使って学習した場合、50ラベルだけで精度は90%以上でる。  

## 先行研究と比べて何がすごい？
特に少ないラベル数でMNISTの精度が出たこと。
GANをクラス分類に使う時のネットワークの工夫がExcellent

## どうやって有効だと検証した?
MNIST, Cifar10, SVHNデータセットを使って、他の手法よりもデータ数が少なくて精度が出ることが証明された。

## 技術の手法や肝は？
Kクラス分類の場合、K+1クラス分類にして、K+1番目はGeneratorが生成した画像が分類されるようにする。  わけではない...
トレーニング時とテスト時で、ネットワーク構成が変わってはいけないことが一般的であるから。  
具体的な説明は、以下サイトがめちゃくちゃ分かりやすい
http://musyoku.github.io/2016/12/23/Improved-Techniques-for-Training-GANs/

## 議論はある？
GANは学習が安定しないことで有名であるので、そこの対策をしておかないとHyperParameterで大きく結果が変動するかも  

## 次に読むべき論文は？
本論文で比較されていた手法の中で一番精度が高かったもの  
`Auxiliary deep generative models`
```
Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary deep generative
models. arXiv preprint arXiv:1602.05473, 2016.
```
