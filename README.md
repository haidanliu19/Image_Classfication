# 논문 코드 구현

## 1. Data set 
   1-1. Cifar10
   
## 2. Model
'''
### 샘플 - LeNetV5
python train.py --config ./configs/model_train.yaml --device 0,1 

### AlexNet
python train.py --config ./configs/model_train_AlexNet.yaml --device 0,1

### VGGNet
python train.py --config ./configs/model_train_VGGNetA.yaml --device 0,1

python train.py --config ./configs/model_train_VGGNetA-LRN.yaml --device 0,1

python train.py --config ./configs/model_train_VGGNetB.yaml --device 0,1

'''
