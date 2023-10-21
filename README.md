# optimizer_comparison

Download bird dataset and extract it to dataset folder
or set path manually

For training:
```bash
cd src
#AdamW
python train.py --opt adamw --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model resnet34 --experiment-name resnet34-adamw
#Lion
python train.py --opt lion --momentum 0.99 --weight-decay 1e-03 --lr 0.0001 --model resnet34 --experiment-name resnet34-lion
```