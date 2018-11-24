# Learning to Track at 100 FPS with Deep Regression Networks
(2016) David Held / Sebastian Thrun  
https://arxiv.org/pdf/1604.01802.pdf  

## どんなもの?（アブストと結論とイントロ）
- GOTURN(Generic Object Tracking Using Regression Networks)
- ニューラルネットを使った一般物体トラッキングで100FPSを達成した初めての事例

## 先行研究と比べて何がすごい？（関連研究）
- Online training ではなく Offline training（先行研究は0.8FPS-15FPSだったが、これは100FPSで動く）
- 特定の物体ではなく、任意の物体をトラッキングできる
- 物体候補のパッチ群に対するスコアづけ（Classificationベース）ではなく、2枚の画像から直接座標を回帰する（Regressionベース）

## どうやって有効だと検証した?（実験）
- 25 videos from the VOT 2014 Tracking Challenge でテストを行った

## 技術の手法や肝は？（マテリアル&メソッド）
- t-1フレームでトラッキングしたい物体をクロップして中心cx,cyに配置して入力1(what to track)
- tフレームでcx,cyの定数倍の領域をクロップして入力2（search region）
- search regionに対してbounding boxの左上と右下の座標を回帰する

## 議論はある？（ディスカッション）
- 物体が何かに遮られるのに弱いので、他の手法と組み合わせるのがいい

## 次に読むべき論文は？（参考文献）
- Fully-Convolutional Siamese Networks for Object Tracking
