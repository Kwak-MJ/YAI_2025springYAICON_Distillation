1. train.py
  Teacher model (ImageNet1k pretrained) -> acc 69.87%
    python train.py --model_job teacher --epochs 100 --learning_rate 0.001 --batch_size 256 --scheduler_step 25
  
  Student model (no pretrained) -> acc 59.22%
    python train.py --model_job student --epochs 50 --learning_rate 0.001 --batch_size 256 --scheduler_step 20

2. train_distillation.py
  Distilled_{Student} (no pretrained) -> acc
    python train_distillation.py --alpha 0.7 --T 2 --epochs 50 --learning_rate 0.001 --batch_size 256 --scheduler_step 20

3. test.py
  Teacher model
    python test.py --model_job teacher

  Student model
    python test.py --model_job student

  Distilled_{Student}
    python test.py --model_job distillation

Teacher = ResNet 50
Student = ResNet 18



실험 결과 정리

1. Teacher model 
    (ImageNet1k pretrained resnet50, 100 epochs, lr=0.001, sched=x0.1, per 25epochs)
    python test.py --model_job teacher
    결과 = Test Loss: 0.00748289, Top-1 test Accuracy: 69.87000000%, Top-5 test Accuracy: 91.17000000%

2. Student model 
    (no pretrained resnet18, 50 epochs, lr=0.001, sched=x0.1, per 20epochs)
    python test.py --model_job student
    결과 = Test Loss: 0.00915756, Top-1 test Accuracy: 59.22000000%, Top-5 test Accuracy: 84.59000000%

3. Distilled_resnet18 
    (same student model, 50 epochs, lr=0.001, sched=x0.1, per 20epochs, T=2, alpha=0.7)
    python test.py --model_job distillation
    결과 = Test Loss: 0.01049371, Top-1 test Accuracy: 60.15000000%, Top-5 test Accuracy: 85.35000000%

4. Distilled_resnet18_KMJ2
    (same student model, 50 epochs, lr=0.001, sched=x0.1, per 20epochs, T=2, alpha scheduling by iteration)
    python test.py --model_job distillation_KMJ2
    결과 = Test Loss: 0.01071865, Top-1 test Accuracy: 60.37000000%, Top-5 test Accuracy: 85.53000000%

5. Distilled_resnet18_KMJ1
    (same student model, 60 epochs, lr=0.001, sched=x0.1, per 20epochs, T=2, alpha scheduling by epoch)
    python test.py --model_job distillation_KMJ1
    결과 = Test Loss: 0.01063989, Top-1 test Accuracy: 60.41000000%, Top-5 test Accuracy: 85.50000000%

6. Distilled_resnet18_KMJ0
    (same student model, 50 epochs, lr=0.001, sched=x0.1, per 20epochs, T=2, alpha scheduling by epoch)
    python test.py --model_job distillation_KMJ0
    결과 = Test Loss: 0.01058444, Top-1 test Accuracy: 60.60000000%, Top-5 test Accuracy: 85.57000000%

7. Distilled_resnet18_LDH
    (beta=0.5, gamma=0)
    python test_LDH.py --model_job distillation_LDH (MODEL_NAME="Distilled_resnet18_LDH")
    결과 = Test Loss: 0.00923391, Top-1 test Accuracy: 63.46000000%, Top-5 test Accuracy: 87.70000000%

8. Distilled_resnet18_LDH2
    (beta=0, gamma=0.2)
    python test_LDH.py --model_job distillation_LDH2 (MODEL_NAME="Distilled_resnet18_LDH2")
    결과 = Test Loss: 0.01046962, Top-1 test Accuracy: 60.49000000%, Top-5 test Accuracy: 85.33000000%

9. Distilled_resnet18_LDH3
    (beta=0, gamma=0.1)
    python test_LDH.py --model_job distillation_LDH3 (MODEL_NAME="Distilled_resnet18_LDH3")
    결과 = Test Loss: 0.01057834, Top-1 test Accuracy: 60.65000000%, Top-5 test Accuracy: 85.51000000%

10. Distilled_resnet18_LDH5 (ce_loss, feature loss=MSELoss 만 사용) 
    (alpha = 0.7)
    python test_LDH.py --model_job distillation_LDH5
    결과 = Test Loss: 0.00746035, Top-1 test Accuracy: 63.17000000%, Top-5 test Accuracy: 87.52000000%

11. Distilled_resnet18_LDH6 (ce_loss, feature loss=MSELoss 사용) 
    (Epoch 기반 loss 가중치 조절)
    python test_LDH.py --model_job distillation_LDH6
    결과 = Test Loss: 0.00675732, Top-1 test Accuracy: 67.32000000%, Top-5 test Accuracy: 89.10000000%

12. Distilled_resnet18_LDH9 (ce_loss, feature loss=mse loss로 사용) 
    (Iteration 기반 loss 가중치 조절)
    python test_LDH.py --model_job distillation_LDH9
    결과 = Test Loss: 0.00686661, Top-1 test Accuracy: 65.92000000%, Top-5 test Accuracy: 89.11000000%

13. Distilled_resnet18_LDH10 (ce_loss, feature loss=mse loss 사용)
    (Iteration 기반, but 첫 epoch만 )
    python test_LDH.py --model_job distillation_LDH10
    결과 = Test Loss: 0.00683499, Top-1 test Accuracy: 66.60000000%, Top-5 test Accuracy: 89.26000000%
