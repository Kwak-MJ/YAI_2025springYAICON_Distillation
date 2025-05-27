import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms


from DataLoader import LoadData
# 사용할 teacher, student 모델로 바꿔서 불러오면 됩니다
from Model import resnet18, resnet50, modelWithFeat


ckp_path = "/root/YAICON/checkpoint"

# test 함수는 수정 금지


def test(model, model_name=None):
    print("================================================")
    print("Test start")
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
    test_loader = LoadData(
        train=False, transform=transform, batch_size=256).get_data()

    # model gpu 설정
    model = model.to(device)

    # optimizer, loss function
    criterion = nn.CrossEntropyLoss()

    # acc
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    # test model
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()

            _, top1_predicted = torch.max(output.data, 1)
            total_samples += y.size(0)
            top1_correct += (top1_predicted == y).sum().item()

            _, top5_predicted = torch.topk(output, 5, dim=1)
            for j in range(y.size(0)):
                if y[j] in top5_predicted[j]:
                    top5_correct += 1

    top1_acc = 100 * top1_correct / total_samples
    top5_acc = 100 * top5_correct / total_samples

    print(
        f"Test Loss: {total_loss/len(test_loader.dataset):.8f}, Top-1 test Accuracy: {top1_acc:.8f}%, Top-5 test Accuracy: {top5_acc:.8f}%")

    print("Test complete")
    print("================================================")


# MODEL_JOB 및 선언하는 모델은 필요시 수정
if __name__ == "__main__":
    ### Change here ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_job', type=str,
                        default='teacher', help='teacher, student, distillation')
    args = parser.parse_args()
    MODEL_JOB = args.model_job
    ######
    MODEL_NAME = None

    if MODEL_JOB == 'teacher':
        model = resnet50()  # teacher
        MODEL_NAME = "resnet50"

        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))

    elif MODEL_JOB == 'student':
        model = resnet18()  # student
        MODEL_NAME = "resnet18"

        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))
    elif MODEL_JOB == 'distillation':
        ### Change here ###
        model = resnet18()  # student
        MODEL_NAME = "Distilled_resnet18"
        ######
        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))
    elif MODEL_JOB == 'distillation_KMJ0':
        ### Change here ###
        model = resnet18()  # student
        MODEL_NAME = "Distilled_resnet18_KMJ0"
        ######
        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))
    elif MODEL_JOB == 'distillation_KMJ1':
        ### Change here ###
        model = resnet18()  # student
        MODEL_NAME = "Distilled_resnet18_KMJ1"
        ######
        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))
    elif MODEL_JOB == 'distillation_KMJ2':
        ### Change here ###
        model = resnet18()  # student
        MODEL_NAME = "Distilled_resnet18_KMJ2"
        ######
        model.load_state_dict(torch.load(ckp_path +
                                         "/{}.pth".format(MODEL_NAME)))
    else:
        raise ValueError(
            "MODEL_JOB can be 'teacher','student','distillation', 'distillation_KMJ1', or 'distillation_KMJ2' only.")

    test(model, MODEL_NAME)
