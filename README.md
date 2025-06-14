# 2025 spring YAICON

Distillation의 loss 가중치인 alpha를 고정하지 않고 변화시키며 학습하는 것을 실험하였다.

기반으로 한 논문은 distillation을 제안한 https://arxiv.org/abs/1503.02531 이다.

주요 아이디어로는 초기 학습 단계에서 distillation loss의 가중을 크게하여 teacher model이 guide를 잘 생성하고 이후 ce loss를 통한 효율적인 학습이 가능할 것이라고 생각하였다.

특히 logit based distllation보다 feature loss based distllation을 사용할 때 신경망 전반에 걸친 regularization 효과가 극대회되어 adaptive distillation의 효과가 더욱 좋을 것이라고 가정하고 실험을 진행하였다.

![image](https://github.com/user-attachments/assets/ff3ff2b1-1c56-473f-ad8e-be6ad107e811)

실험 결과 실제로 feature loss + adaptive distillation을 사용한 Ada Feature Loss가 기존 논문의 방법보다 눈에 띄게 성능이 향상되며 teacher model과도 비교할만한 결과를 보여주는 것을 확인하였다.


Teacher = ResNet 50
Student = ResNet 18

### 학습 방법
1. train.py Teacher model (ImageNet1k pretrained) -> acc 69.87%
   python train.py --model_job teacher --epochs 100 --learning_rate 0.001 --batch_size 256 --scheduler_step 25
2. Student model (no pretrained) -> acc 59.22%
   python train.py --model_job student --epochs 50 --learning_rate 0.001 --batch_size 256 --scheduler_step 20
3. train_distillation.py Distilled_resnet18 (no pretrained)
   python train_distillation.py --alpha 0.7 --T 2 --epochs 50 --learning_rate 0.001 --batch_size 256 --scheduler_step 20
4. train_distillation_{NAME}.py Distilled_resnet18_{NAME} (no pretrained)
   python train_distillation_{NAME}.py --alpha 0.7 --T 2 --epochs 50 --learning_rate 0.001 --batch_size 256 --scheduler_step 20

### 모델 다운로드
https://drive.google.com/file/d/1p4KQN7lNXGBpKlI3TNBo2xlAAiFIPLX9/view?usp=sharing

checkpoint \
 L {MODEL}.pth \
 L {MODEL2}.pth \
 L ... 

### 테스트 방법
1. Teacher model
   python test.py --model_job teacher

2. Student model 
   python test.py --model_job student

3. Distilled_resnet18_{NAME}
   python test.py --model_job distillation_{NAME}

