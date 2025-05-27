# 2025 spring YAICON

Distillation의 loss 가중치인 alpha를 고정하지 않고 변화시키며 학습하는 것을 실험하였습니다.

주요 아이디어로는 초기 학습 단계에서 distillation loss의 가중을 크게하여 teacher model이 guide를 잘 생성하고 이후 ce loss를 통한 효율적인 학습이 가능할 것이라고 생각하였다.

특히 이러한 adaptive distillation loss를 feature based distillation과 함께 사용했을때 spatial information을 통해 guide의 효과가 증폭될 것이라고 생각하고 실험하였다.


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

### 테스트 방법
1. Teacher model
   python test.py --model_job teacher

2. Student model 
   python test.py --model_job student

3. Distilled_resnet18_{NAME}
   python test.py --model_job distillation_{NAME}

