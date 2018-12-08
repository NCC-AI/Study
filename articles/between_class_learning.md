# Between-class Learning for Image Classification
(2018) Yuji Tokozume, Yoshitaka Ushiku, Tatsuya Harada  
https://arxiv.org/pdf/1711.10284.pdf

## どんなもの?
画像をランダムな比率で混ぜ合わせて、クラス分類をその比率の回帰にする。
![mix_img](https://github.com/NCC-AI/Study/blob/images/BetweenClass/mix.png)

## 先行研究と比べて何がすごい？
- Between Class Learningの手法と、その妥当性の提案。
    - なるべく分類は、モデルのロスの収束が、画像の分布の分離になっているようにする。

## どうやって有効だと検証した?
Cifar10, Cifar100でもエラー率が下がってる。

## 技術の手法や肝は？
![img](https://github.com/NCC-AI/Study/blob/images/BetweenClass/betweencls.png)  

## 議論はある？
- Limitationはある？　いつでも必ず使えるのか？

## 次に読むべき論文は？
原田先生論文。Domain Adaptationとか
