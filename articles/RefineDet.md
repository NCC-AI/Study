# Single-Shot Refinement Neural Network for Object Detection
(2018) Shifeng Zhang / Longyin Wen  
https://arxiv.org/pdf/1711.06897.pdf

## どんなもの?（アブストと結論とイントロ）
![result](https://github.com/NCC-AI/Study/blob/images/RefineDet/RefineDet_result.png)
- 一般的に物体検出は2ステージアプローチ（R-CNN）が精度、1ステージアプローチ（SSD）が速度において優れている
- 本論文のRefineDetは2ステージアプローチより高精度で、1ステージアプローチの速度を維持した検出器

## 先行研究と比べて何がすごい？（関連研究）
- 両アプローチのメリットを引き継ぎ、デメリットを少なくした
- 2 stage approachのメリット
  - クラスの不均衡問題に対処できる
  - 1st stageでボックスパラメータの粗い予想、2nd stageでその微調整を行うことで精度が上がる
  - 1st stageで物体があるかないかの2値分類、2nd stageで多クラス分類を行うことで精度が上がる

## どうやって有効だと検証した?（実験）
- PASCALVOC2007,PASCALVOC2012,MSCOCOで、FPS,mAPを評価した

## 技術の手法や肝は？（マテリアル&メソッド）
![model](https://github.com/NCC-AI/Study/blob/images/RefineDet/RefineDet_model.png)
- Anchor Rifinement Module(ARM)
  - 分類時の検索スペースを減らすためnegative anchorを減らす
  - anchor boxの位置とサイズに対しての最適な初期化と、物体があるかないかの2クラス分類を行う
- Object Detection Module(ODM)
  - ARMの出力であるRefinement Anchorを入力として利用する
  - anchor boxの最終調整と多クラス分類を行う
- Transfer Connecting Block(TCB)
  - ARMで抽出した情報をODMに受け渡す
  - 深い層の特徴を浅い層に還元する


## 議論はある？（ディスカッション）
- まだ小さい物体の認識が弱い、現状では画像サイズを320×320から512×512に拡大することでしか解決できない


## 次に読むべき論文は？（参考文献）
- [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
  - 2018年4月に発表されたYOLO系の最新版
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
  - 1 stage approachの精度の低さがデータセットの偏りに起因している話

