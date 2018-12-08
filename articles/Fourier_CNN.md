
# FCNN: Fourier Convolutional Neural Networks
(2017) Harry Pratt / Bryan Williams (University of Liverpool)  
http://ecmlpkdd2017.ijs.si/papers/paperID11.pdf

## どんなもの?
- 概要
  - CNNのConvolutionを周波数空間で行うようなネットワークを構築
  - Classificationタスクの実行時間を比べたら大きな画像サイズでは提案手法のほうが最大40倍高速だった

- イントロ
  - CNNが発展しているが、サイズが大きなデータ（大きな画像など）では計算速度が課題  
  - CNNの重さの原因はConvolutionで、Sliding-windowで計算することが原因
  - Fourier Convolutional Neural Networksを用いることで高速なトレーニング＆有効性に影響を与えない
  - Fourier変換を画像解析や信号処理の文脈でNerual Networkに使用している研究はあるが、まだ初期段階でありその貢献は小さい  
  ![image](https://user-images.githubusercontent.com/12442472/48743358-25efd100-eca6-11e8-9214-97e3e107d810.png)  

## 先行研究と比べて何がすごい？
- Fourier領域で処理することでConvolutionを高速化する研究はあるが、処理中で空間領域とFourier領域を行き来するため無駄が生じている
- またその他の先行研究でFourier領域でダウンサンプリングすることで空間情報をなるべく保持したまま収束性を向上させる手法が提案されている
- 提案手法では処理前にFourier変換を施し、逆フーリエ変換はしないまま処理を最後まで行う


## どうやって有効だと検証した?
- 概要
  - 単純〜複雑なデータセットでAccuracyと速度を比較
  - ネットワークの層数は同じにした
  - MNIST (28 x 28のグレースケール画像、10 Class分類問題)
  - Cifar-10 (32 x 32のRGB画像、10 Class分類問題)
  - Kaggleの超音波画像データセット（元画像をダウンサンプリング、5 Class分類問題）
- ネットワークについて
  - 処理がシンプルなのでAlexNetをベースに選択（渡部：今はResNetが性能良いとされているが、あえてAlexNetを選んでいると思われる）
  - Convolution Kernelは周波数領域で表現、PoolingはFourier領域で間引くことで代用（異なる処理になるが十分動く）  
  ![image](https://user-images.githubusercontent.com/12442472/48743260-b24dc400-eca5-11e8-8c6c-d4835a4e0757.png)  
  ![image](https://user-images.githubusercontent.com/12442472/48743282-cee9fc00-eca5-11e8-8a87-6b1c4563a63b.png)  
  - DropoutとDense結合はそのままFourier領域で実装
  - バックエンドをTheanoにしてKerasで実装
  - FFTの層はTheanoで実装
- 実験結果
![image](https://user-images.githubusercontent.com/12442472/48743331-022c8b00-eca6-11e8-9582-e9cdc63dda64.png)  
![image](https://user-images.githubusercontent.com/12442472/48743350-1a040f00-eca6-11e8-8de4-c16284c4c8a3.png)  
  - MNISTはAccuracy下がった、Cifar-10では上がった
  - 処理速度は画像サイズによるが小さい場合は同じ速さ、画像サイズに対して指数関数的に差がつく



## 技術の手法や肝は？
- 空間領域でのConvolutionはFourier領域でのHadamard Production（要素ごとの積）になるので計算コストが低くなる
- FFTとIFFTを行き来しないのでコスト削減になった
- 空間領域のKernelとくらべて周波数領域のKernelは自由度が大きいので表現力UP
- (渡部) ↑それはそうだけどパラメータ数増えるからメモリ圧迫しない？大きなモデルだと弱点になるのでは？


## 議論はある？
- MNISTで精度落ちてるのは画像サイズが小さすぎてFFTするときに情報が欠損するから
- Cifar-10ではFCNNがAccuracy上回った、これはKernelがFourier空間だと表現力が増すから
- (渡部) ↑それはパラメータ数が増えたからでは？同じ層数で比較するのも良いけど、同じパラメータ数で比較したらどうなる？
- ネットワークのアーキテクチャに依存せず使えるので、いろいろなタスクで使われると良い
- (渡部) 空間CNNでKernelサイズが大きいと深い層での受容野が大きくなるので画像の文脈を考慮しやすいとの論文を見たことがあるが、
このFCNNは空間領域のKernelサイズは実質任意ということになるので文脈を考慮した処理ができるのではないか？

## 次に読むべき論文は？
前にFFTで訓練すると早くなるって言ってた人のやつ  
Yann LeCun Michael Mathieu, Mikael Henaff. Fast training of convolutional networks
 through ffts, 2014. 3  
その他は、これを周波数空間の情報を取り出すために使ってる人の論文を探したい。


















