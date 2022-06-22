from utils.dataloader_image_classification import ImageTransform, make_datapath_list, HymenopteraDataset
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

# Datasetを作成する
size = 2242
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train'
)

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val'
)

# DataLoaderを作成する
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

# 辞書オブジェクトにまとめる
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

# ネットワークモデルを作成
net = models.vgg16(pretrained=True)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードの設定
net.train()

print('ネットワーク設定完了')

# 損失関数を定義
loss_func = nn.CrossEntropyLoss()

# 最適化手法を設定
# ファインチューニングでは、全層のパラメータを学習できるようにoptimizerを設定する。
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 学習させる層のパラメータ名を設定
update_param_name_1 = ['features']
update_param_name_2 = ['classifier.0.weight',
                       'classifier.0.bias',
                       'classifier.3.weight',
                       'classifier.3.bias']
update_param_name_3 = ['classifier.6.weight',
                       'classifier.6.bias']

# パラメータごとに各リストに格納する
for name, param in net.named_parameters():
    if name in update_param_name_1:
        param.required_grad = True
        params_to_update_1.append(param)
        print(f'pramas_to_update_1に格納: {name}')

    elif name in update_param_name_2:
        param.required_grad = True
        params_to_update_2.append(param)
        print(f'pramas_to_update_2に格納: {name}')

    elif name in update_param_name_3:
        param.required_grad = True
        params_to_update_3.append(param)
        print(f'pramas_to_update_3に格納: {name}')

    else:
        param.requires_grad = False
        print(f'勾配計算なし: {name}')

# 最適化関数を定義
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3},
])


def train_model(net, dataloaders_dict, loss_func, optimizer, num_epochs):

    # 初期設定
    # GPU が使えるか確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} / {num_epochs}')
        print(f'----------------------')

        # epochごとに訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)  # 第一戻り値には、Tensorの値、第二戻り値には最大の値のインデックスを返す

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

num_epochs = 4
train_model(net, dataloaders_dict, loss_func, optimizer, num_epochs=num_epochs)