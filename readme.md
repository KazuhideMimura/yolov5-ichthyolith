# yolov5-ichthyolith
![Visitors](https://visitor-badge.glitch.me/badge?page_id=KazuhideMimura/yolov5-ichthyolith&left_color=gray&right_color=blue)

**yolov5-ichthyolith** is an application of yolov5 for detecting microfossil fish teeth called ichthyolith.

code originality: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## Citation
```
@software{yolov5-ichthyolith,
  author = {Kazuhide Mimura},
  month = {5},
  title = {{Automated detection of microfossils by combining YOLO-v5 and EfficientNet}},
  url = {https://github.com/KazuhideMimura/yolov5-ichthyolith/},
  year = {2022}
}
```

## What is new?
Combining detection and classification model is useful when many FPs are contained in images.

In original yolo-v5, number of classes for 1st (detection) model and 2nd (classification) model should be the same.

However, enabling number of classes for 2nd models may benefit many object detection problem using yolov5. 
See [ultralytics/yolov5/issues/7429](https://github.com/ultralytics/yolov5/issues/7429) for detail.

For this purpose, detect.py and classifier.py were modified so that second model can classifiy objects into larger number of classes than that of 1st models.

The code is also designed to try "fine- and coarse- grain labeling" reported by [Chen et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8637482).

<details><summary>
日本語
</summary><div>

物体検出モデルで検出したものを，画像分類モデルで再分類することは，False Positive が多い検出問題では特に有効だと考えられています．

物体検出モデルとして現在広く用いられている YOLO-v5 にも，EfficientNet 等の画像分類モデルを用いて再分類する機能が実装されていましたが，
「物体検出モデルのクラス数」と「画像分類モデルのクラス数」が同じである必要がありました．
  
本プロジェクトでは，画像分類モデルのクラス数を物体検出モデルのクラス数と無関係に決定できるようにプログラムコードの変更を行いました．
これは，以下の２点でメリットがあると考えられます．
  
(1) 学習の手間の削減
画像分類モデルの学習や教師データの準備は，物体検出モデルのそれらと比較して容易です．
このため，「物体検出モデルには輪郭の抽出のみを学習させ，クラスの判定は画像分類モデルで訓練する」といった活用方法が考えられます．
  
(2) 分類精度の向上
画像分類モデルは，ラフなラベル（例えば「動物」）で学習するよりも詳細なラベル（例えば「犬，猫，…」）で学習する方が高精度であるということが，
[Chen et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8637482) で報告されています．

本プロジェクトでも，このことを試すことができるように，「詳細に分類したラベルからラフなラベルに戻す」機能を実装しています
  
</div></details>

## Command and detection examples
#### train detection (1st) model
`python train.py --img 800 --batch 16 --epoch 80 --data ichthyolith_detection.yaml --weights yolov5l.pt --name 20220510_model1`

#### train classification (2nd) model
`python classifier.py --model efficientnet_b0 --data path/to/dataset --project second_classifier --name 20220510_model2 --epochs 15 --img 224`

#### detect using only 1st model
`python detect.py --source path/to/detection/directory --weights runs/train/20220510_model1/weights/best.pt --img 800`
<br>
<img src="/images_for_github/1_first.jpg" width="500">
<br>

#### detect using both 1st and 2nd model
`python detect.py --source path/to/detection/directory --weights runs/train/20220510_model1/weights/best.pt --second second_classifier/20220510_model2/weights/best.pt --img 800`
<br>
<img src="/images_for_github/2_second.jpg" width="500">
<br>

#### detect using both 1st and 2nd model (convert to coarse labels)
`python detect.py --source path/to/detection/directory --weights runs/train/20220510_model1/weights/best.pt --second second_classifier/20220510_model2/weights/best.pt --return-coarse --img 800`
<br>
<img src="/images_for_github/3_re-coarsed.jpg" width="500">
<br>

## Reference
[ultralytics/yolov5](https://github.com/ultralytics/yolov5)

also see: [ultralytics/yolov5/issues/7429](https://github.com/ultralytics/yolov5/issues/7429)

## log
2022/6/7: started counting visitors

2022/6/4: Added some program codes that were missing in utils/~.

2022/5/14: enabled to return to coarse labels & added some detection images to readme.md

2022/5/10: released

## todo
- find some images where FPs can be reduced using 2nd models
