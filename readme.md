**yolov5-ichthyolith** is an application of yolov5 for detecting microfossil fish teeth called ichthyolith.

code originality: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## What is new?
(in preparation)

## Command example
#### train detection (1st) model
`python train.py --img 800 --batch 16 --epoch 80 --data ichthyolith_detection.yaml --weights yolov5l.pt --name 20220510_model1`

#### train classification (2nd) model
`python classifier.py --model efficientnet_b0 --data path/to/dataset --project second_classifier --name 20220510_model2 --epochs 15 --img 224`

#### detect using only 1st model
`python detect.py --source path/to/detection/directory --weights runs/train/20220510_model1/weights/best.pt --img 800`

#### detect both 1st and 2nd model
`python detect.py --source path/to/detection/directory --weights runs/train/20220510_model1/weights/best.pt --second second_classifier/20220510_model2/weights/best.pt --img 800`

## Reference
[ultralytics/yolov5](https://github.com/ultralytics/yolov5)

also see: [ultralytics/yolov5/issues/7429](https://github.com/ultralytics/yolov5/issues/7429)

## log
2022/5/10: released

## todo
- update val.py for 2nd classification model
