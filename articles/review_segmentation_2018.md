# Methods and datasets on semantic segmentation: A review  
(2018) Hongshan Yua / Zhengeng Yang  
https://www.sciencedirect.com/science/article/pii/S0925231218304077  
（大学からじゃないと見れないかも）  

### Part 1
10分で全体を説明するのは厳しいので主に論文のうちで表現学習の手法を扱った部分について説明する。    

## どんなもの?
- 全体について  
  - Semantic Segmentationについて、① 手動設計特徴量, ② 表現学習, ③ 半教師あり学習 について最近の進歩をまとめた  
  - 新しいデータセットについてもまとめた  
  - モデルごとに長所と短所を比較した  
  - 今後の研究が発展するであろう分野について論じた  

  
## 表現学習を用いた手法について  
![image](https://user-images.githubusercontent.com/12442472/48664565-d95c9800-eae3-11e8-89e2-a601ec6bf8bc.png)  
- ナイーブな手法
  - 一番単純な手法
  - CNNを使って画素ごとにラベルを推定する    
  - 固定サイズの領域に対してラベリングを行い、それをすべての場所で行う
  - 精度の良い推定のためには切り出し領域サイズを大きくしたいが、計算コスト増大  
  - 入力画像のスケールを変更したものを複数用意し、それぞれにCNN適用で広い範囲の情報を見たことにする  
  - それでも全部の画素を見るのは計算コスト的に厳しい  
  - 無駄なのは推定したピクセルのすぐ横のピクセルについては計算がほぼ同じなので冗長なことをしていること  
  - Region　Proposalと組み合わせた手法などが提案された  
  - 2015年には特徴マップをアップサンプリングしてセグメンテーションを行う手法も提案された  
- FCN
  - Shift & Stitch
    - 複数の適当な間隔でシフトさせたのちに低解像度に落とした画像をCNNで処理、でも結局シフトさせた画像ごとに処理なので計算コスト大
    - より効率的な方法は、荒い解像度の予測をアップサンプリングすること（UP_FCN）、その方法はバイリニアの補間みたいな簡単なものでも、デコンボリューションのような手法でもいい  
  - アップサンプリングを用いる手法
  ![image](https://user-images.githubusercontent.com/12442472/48664561-cd70d600-eae3-11e8-96d2-74dd05df0acd.png)  
  
    - UP_FCNはEnd-to-End（たとえば、特徴表現だけCNNで分類器は別、というのはEnd-to-Endでない）
    - ただしあまりに低解像度な特徴マップのアップサンプリングは情報の損失が大きすぎて微妙  
    - とはいえダウンサンプリングを減らしすぎると最後の層（一番解像度が低い部分）の受容野が小さくなりすぎて微妙
    - SegNetはデコーダ部分を階層構造にするなどしてこの問題に対応  
    - Convolutionのカーネルの構造をDilatedにしたり、aturalにすることで解像度を解像度高めで受容野も大きめ、というのをやっている例もある  
    - しかしながらカーネルの構造をDilatedにしたいするとlow-level（細かい構造だと思う）の情報がとれない
    - 現実的には最後の層だけに注目せずに中間層をSkipwise Connectionで利用する方法が良い
    - 実際にRefine Net, RDF Net, Lable Refinementなどの最近の手法でもSkipwise Connectionの考え方が使用されている  
    
  - 受容野は理論的には大きいから入力全体の情報を使っている？ → No
    - 理論的な受容野が大きくても有効な受容野のサイズは小さいことが報告されている、すなわちGlobalなContextが活用できていない？ 
    - （コメント：↑解剖構造とか見ようと思ったら考えなくちゃいけないような気がする）
    - Pyramid Pooling（プーリングの範囲を複数作る）ことで受容野を拡大することができた  
    - 同様のアプローチでより効率的な方法はatrous spatial pyramid pooling(ASPP)
    - Contextを活用するという意味でConditional Random Field （CRF)を使用した方法がある
    - CRFは入力画像のテクスチャ情報から特徴をK-meansなどでいくつか分類し、その位置関係や色などから文脈を推定する方法（らしい、よくわかってない）
    - CRF+FCNは強力だがネットワークが複雑になったり訓練が大変になったりする
    - CRFをCNNで近似する方法などが研究されている
    
  - 今はどんなことが研究されているのか？
  ![image](https://user-images.githubusercontent.com/12442472/48664566-dcf01f00-eae3-11e8-8cb0-adc6ce291ddd.png)
    - Skip Connectionを使用したUP_FCNとatrous UP_FCNがベース
    - より強力なCNNモデル（ResNetとか）
    - これからの課題：より高解像度な特徴量、いまはメモリ的に入力の1/8が限界。
    
   
## 次に読むべき論文は？
今までの流れがわかったので、リアルタイム性があるネットワークで性能いいのはどれか検討するために最新の手法を見てみる
[135] S. Rota Bulo, G. Neuhold, P. Kontschieder, Loss max-pooling for semantic image
segmentation, in: Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2017, pp. 2126–2135.




