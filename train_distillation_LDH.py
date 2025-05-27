# train_distillation 과 거의 동일합니다.

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from DataLoader import LoadData

### Change model import here ###
from Model import resnet18, resnet50, modelWithFeat
######

ckp_path = "/root/YAICON/checkpoint"

# Logit based distillation model
# teacher는 pre-trained model
# student는 non-trained model를 인자로 받음


def train_distillation(teacher, student, alpha=0.7, T=2, epochs=100, learning_rate=0.001, batch_size=256, model_name=None, scheduler_step=20):
    print("================================================")
    print("Train start")
    print("Model: {}".format(model_name))

    # device 설정
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # transform 설정 (getData 자체적으로도 존재해서 따로 주석처리 해도 문제는 없음)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[
                             0.2675, 0.2565, 0.2761])
    ])

    # LoadData().get_data()로 train_loader 불러오기
    train_loader = LoadData(
        train=True, transform=transform, batch_size=batch_size).get_data()

    # model gpu 설정
    teacher = teacher.to(device)
    student = student.to(device)

    # optimizer, loss function, hyperparameter
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
    beta = 0.5  # 임시로 hyperparameter 설정, feature loss
    gamma = 0.05  # new loss

    # train model
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            for param in teacher.parameters():
                param.requires_grad = False

            with torch.no_grad():
                teacher_logit, teacher_feat = teacher(x)
            student_logit, student_feat = student(x)

            ### distillation loss, 핵심 아이디어 ###
            soft_teacher_prob = nn.functional.softmax(teacher_logit/T, dim=-1)
            soft_student_prob = nn.functional.log_softmax(
                student_logit/T, dim=-1)

            soft_loss = (T**2) / soft_student_prob.size()[0] * torch.sum(
                soft_teacher_prob * (soft_teacher_prob.log() - soft_student_prob))

            student_loss = criterion(student_logit, y)
            ###

            # feature loss
            feature_loss = nn.functional.mse_loss(teacher_feat, student_feat)
            ###

            # new loss
            s2t_logit = teacher.fc(teacher.avgpool(
                student_feat).view(x.shape[0], -1))
            s2t_loss = criterion(s2t_logit, y)
            ###

            loss = alpha * student_loss + beta * s2t_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.8f}")

    # 모델 저장
    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)
    torch.save(student.state_dict(), ckp_path +
               "/{}.pth".format(model_name))
    print("Model saved")
    print("Train complete")
    print("================================================")


# Model 이름 필요시 수정
# epoch와 learning_rate 필요시 수정
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='alpha for distillation')
    parser.add_argument('--T', type=float, default=2,
                        help='temperature for distillation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--scheduler_step', type=int, default=20,
                        help='scheduler step size')
    args = parser.parse_args()

    ALPHA = args.alpha  # 변경 가능
    T = args.T  # 변경 가능
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    SCHEDULER_STEP = args.scheduler_step
    MODEL_NAME = "Distilled_resnet18_LDH4"

    ### Change teacher model here ###
    teacher = resnet50()
    teacher.load_state_dict(torch.load(ckp_path +
                                       "/{}.pth".format("resnet50")))
    teacher = modelWithFeat(teacher, isStudent=False)
    student = modelWithFeat(resnet18(), isStudent=True)
    ######

    # train model
    train_distillation(teacher, student, ALPHA, T,
                       EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_NAME, SCHEDULER_STEP)
