import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from DataLoader import LoadData

### Change model import here ###
from Model import resnet18, resnet50
######

ckp_path = "/root/YAICON/checkpoint"


def train(model, epochs=100, learning_rate=0.001, batch_size=256, model_name=None, scheduler_step=20):
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
    model = model.to(device)

    # optimizer, loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=0.1)

    # train model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.8f}")

    # 모델 저장
    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)

    torch.save(model.state_dict(), ckp_path +
               "/{}.pth".format(model_name))
    print("Model saved")
    print("Train complete")
    print("================================================")


# Model 이름 필요시 수정
# epoch와 learning_rate 필요시 수정
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_job', type=str, default='teacher',
                        help='teacher or student')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--scheduler_step', type=int, default=20,
                        help='scheduler step size')

    args = parser.parse_args()
    MODEL_JOB = args.model_job
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    SCHEDULER_STEP = args.scheduler_step
    ######

    MODEL_NAME = None

    if MODEL_JOB == 'teacher':
        model = resnet50()
        MODEL_NAME = "resnet50"
    elif MODEL_JOB == 'student':
        model = resnet18()
        MODEL_NAME = "resnet18"
    else:
        raise ValueError("MODEL_JOB should be 'teacher' or 'student'")

    # train model
    train(model, EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_NAME, SCHEDULER_STEP)
