# Dynamic Routing Between Capsules
(2017.8) Sara Sabour/ Nicholas Frosst/ Geoffrey E Hinton 
https://arxiv.org/abs/1710.09829


## どんなもの?
1. CNNでの画像認識は人間と同じとは言い難く、"route"を考慮できていない →CNNではプーリングの副作用で本質的にパーツの位置関係や姿勢などの認識ができていない
1. 従来では重みと入力の内積というスカラーの集合であった層を、ベクトルの集合（Capsules）とすることで姿勢などの情報を使用できるようにした
1. 層間のCapsulesの結合（routing）を学習できるようにした．
![image](https://user-images.githubusercontent.com/12442472/44302321-f6f15880-a360-11e8-9194-d7c239b34794.png)
![face_position](https://user-images.githubusercontent.com/12442472/44302109-a6c4c700-a35d-11e8-9134-d668561d02d4.png)
![image](https://user-images.githubusercontent.com/12442472/44302244-9f9eb880-a35f-11e8-90e7-3698634ed686.png)

## 先行研究と比べて何がすごい？
1. 空間の位置情報を保持した学習ができる
 （→データ拡張が不要あるいは少なくても良い）
1. 誤差逆伝搬を使わない Dynamic routingという手法で重み更新

## どうやって有効だと検証した?
1. 実際にCapsulesを用いた3層のネットワーク（CapsNet）を構築した．
1. CapsNetはデータ拡張なしにMNISTのテストエラー0.25%を達成した．
1. 複数の手書き文字を重ね合わせたMultiMNISTデータセットで従来よりも高性能達成
![image](https://user-images.githubusercontent.com/12442472/44302231-72520a80-a35f-11e8-82dd-6984de567f03.png)
![image](https://user-images.githubusercontent.com/12442472/44302238-8138bd00-a35f-11e8-911b-1bc95dbdd8e1.png)

## 技術の手法や肝は？
1. 各層で保持する値をスカラでなくベクトルにした
1. ベクトルに対してaffine変換を施す演算を追加したことで空間的な情報の調整ができる
1. 各層での演算に内積でベクトル同士の関係性を測るステップを導入することで、どの入力Capsuleが出力Capsuleに寄与するか計算できる
![face_position](https://user-images.githubusercontent.com/12442472/44302218-33bc5000-a35f-11e8-8d06-62819936b41f.png)
![image](https://user-images.githubusercontent.com/12442472/44302274-379ca200-a360-11e8-88d9-c445172b973b.png)


## 議論はある？
1. 生物学的に正しい構造
1. 重ね合わせMNISTで効果が出ているのもその証拠
1. 位置情報を従来の文脈で入れようとすると指数関数的計算量増加、でもCapsuleなら変換行列のおかげで大丈夫
？1. （自分の意見）Capsuleのコンセプト実証にはなっているがこれを起点に様々なCapsule構造が発展していくのでは？

## 次に読むべき論文は？
Capsuleのアイデアの元となっている2011年の論文
Geoffrey E Hinton, Alex Krizhevsky, and Sida D Wang. Transforming auto-encoders. In International
Conference on Artificial Neural Networks, pages 44–51. Springer, 2011.


## このまとめにおける参考文献
https://kndrck.co/posts/capsule_networks_explained/
https://qiita.com/hiyoko9t/items/f426cba38b6ca1a7aa2b
https://mosko.tokyo/post/on-capusels/
