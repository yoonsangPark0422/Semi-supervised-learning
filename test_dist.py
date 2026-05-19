

import torch
from torchvision import datasets, transforms
from collections import Counter

from models.wideresnet import WideResNet

model = WideResNet(
    num_classes=10,
    depth=28,
    widen_factor=2,
    drop_rate=0.0
)
# 2. checkpoint 불러오기
checkpoint = torch.load('results/cifar10_imb10/model_best.pth.tar')

# 3. weight만 모델에 넣기
model.load_state_dict(checkpoint['state_dict'])

model.cuda()
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)

counter = Counter()

with torch.no_grad():
    for x, _ in loader:
        x = x.cuda()
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        counter.update(preds.cpu().numpy())

print(counter)

