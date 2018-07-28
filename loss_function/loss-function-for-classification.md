# クロスエントロピーと二乗誤差の違い

- クロスエントロピー  
名前の通り、２つの確率分布(エントロピー)を比較して、分布が同じになるようにするから、カテゴリー分類で、それぞれのクラスの確率分布を最適化できる。
- 二乗誤差  
絶対量を比較する。画像同士の比較は、分布よりも各ピクセルのRGBの絶対量が目的の画像と同じになるようにしなくてはいけない。

- [x] Usage 1
```
model.compile(loss='mean_squared_error', optimizer='sgd')
```
- [x] Usage 2
```
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```
