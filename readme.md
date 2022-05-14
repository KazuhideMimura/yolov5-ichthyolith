**yolov5-ichthyolith** is an application of yolov5 for detecting microfossil fish teeth called ichthyolith.

code originality: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## What is new?
Combining detection and classification model is useful when many FPs are contained in images.

In original yolo-v5, number of classes for 1st (detection) model and 2nd (classification) model should be the same.

However, enabling number of classes for 2nd models may benefit many object detection problem using yolov5. 
See [ultralytics/yolov5/issues/7429](https://github.com/ultralytics/yolov5/issues/7429) for detail.

For this purpose, detect.py and classifier.py were modified so that second model can classifiy objects into larger number of classes than that of 1st models.

The code is also designed to try "fine- and coarse- grain labeling" reported by [Chen et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8637482).

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
2022/5/14: enabled to return to coarse labels & added some detection images to readme.md

2022/5/10: released

## todo
- find some images where FPs can be reduced using 2nd models
