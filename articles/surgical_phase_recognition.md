# Surgical phase modelling in minimal invasive surgery
(2018) F. C. Meeuwsen, F. van LuynM. D. BlikkendaalF. W. JansenJ. J. van den Dobbelsteen
https://link.springer.com/article/10.1007%2Fs00464-018-6417-4

## どんなもの?
腹腔鏡下胆嚢摘出手術の工程解析を行って、手術の終了時刻を予測するモデルを作った。

- 腹腔鏡は一番頻度も高くて、スタンダードな手術(60万件/year in U.S.)
- 腹腔鏡手術は手術時間に差がある(between 98 and 214 min)
- 手術を効率化することはコスト面で大事
- いろんな要因(術者のスキル、患者の状態)で手術時間は変化するけど、普通は平均時間で予定を立てる
- 予定より早く終わる=>手術室の有効活用できて無い, 予定より遅く終わる=>手術が延期やキャンセルになる

## 先行研究と比べて何がすごい？
- ドアの開閉=>術具の使用に注目
- 工程分類が10工程に細分化している

## どうやって有効だと検証した?
10クロスバリデーションを行って、Accuracyは77%
手術の終了時刻予測は、平均16 ± 13 minの誤差だった。

## 技術の手法や肝は？
three cameras and four audio signals
術具の使用を監視。
40症例を、手作業でアノテーションして10クラス分類した  
Randam  Forest

## 議論はある？
3つの設置されたビデオ=>術具の使用を人が確認、タグ付け=>Random Forestで工程分類と終了時刻予測

## 次に読むべき論文は？
