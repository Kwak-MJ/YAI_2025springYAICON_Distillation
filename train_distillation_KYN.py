import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from DataLoader import LoadData
from Model import resnet18, resnet50, modelWithFeat

ckp_path = "/root/YAICON/checkpoint"

ACTIVE_KNOWLEDGE = ["logit", "feature", "similarity"]  

class MSEFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, feat_s, feat_t):
        return self.mse(feat_s, feat_t)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, feat_s, feat_t):
        if len(feat_s.shape) > 2:
            feat_s = torch.flatten(feat_s, start_dim=1)
            feat_t = torch.flatten(feat_t, start_dim=1)
        target = torch.ones(feat_s.shape[0]).to(feat_s.device)
        return self.cos(feat_s, feat_t, target)

def train_distillation(teacher, student, alpha=0.7, T=2, epochs=100, learning_rate=0.001, batch_size=256, model_name=None, scheduler_step=20):
    print("================================================")
    print("Train start")
    print("Model: {}".format(model_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    train_loader = LoadData(train=True, transform=transform, batch_size=batch_size).get_data()

    teacher = teacher.to(device)
    student = student.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=0.1)

    feature_loss_fn = MSEFeatureLoss()
    similarity_loss_fn = CosineSimilarityLoss()

    teacher.eval()
    student.train()

    beta = 0.5
    gamma = 0.2
    delta = 0.3  # similarity loss weight

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

            ce_loss = criterion(student_logit, y)

            kd_loss = 0
            if "logit" in ACTIVE_KNOWLEDGE:
                soft_teacher_prob = nn.functional.softmax(teacher_logit/T, dim=-1)
                soft_student_prob = nn.functional.log_softmax(student_logit/T, dim=-1)
                kd_loss = (T**2) * torch.nn.KLDivLoss(reduction="batchmean")(soft_student_prob, soft_teacher_prob)

            feat_loss = 0
            if "feature" in ACTIVE_KNOWLEDGE:
                feat_loss = feature_loss_fn(student_feat, teacher_feat.detach())

            sim_loss = 0
            if "similarity" in ACTIVE_KNOWLEDGE:
                sim_loss = similarity_loss_fn(student_feat, teacher_feat.detach())

            s2t_logit = teacher.module.fc(teacher.module.avgpool(student_feat).view(x.size(0), -1)) if hasattr(teacher, 'module') else teacher.fc(teacher.avgpool(student_feat).view(x.size(0), -1))
            s2t_loss = criterion(s2t_logit, y)

            loss = alpha * ce_loss + (1 - alpha) * kd_loss + beta * feat_loss + gamma * s2t_loss + delta * sim_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.6f}")

    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)
    torch.save(student.state_dict(), os.path.join(ckp_path, f"{model_name}.pth"))
    print("Model saved")
    print("Train complete")
    print("================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--T', type=float, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--scheduler_step', type=int, default=20)
    parser.add_argument('--knowledges', nargs='+', default=['logit'])
    args = parser.parse_args()

    ALPHA = args.alpha
    T = args.T
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    SCHEDULER_STEP = args.scheduler_step
    ACTIVE_KNOWLEDGE = args.knowledges

    teacher_raw = resnet50()
    teacher_raw.load_state_dict(torch.load(os.path.join(ckp_path, "resnet50.pth")))
    student_raw = resnet18()

    teacher = modelWithFeat(teacher_raw, isStudent=False)
    student = modelWithFeat(student_raw, isStudent=True)

    MODEL_NAME = "Distilled_resnet18_KYN"

    train_distillation(teacher, student, ALPHA, T, EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_NAME, SCHEDULER_STEP)
