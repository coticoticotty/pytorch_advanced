import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

# 学習済のVGG-16モデルをロード
# 初めて実行する際は、学習済みパラメータをダウンロードするため、実行に時間がかかる

# VGG-16モデルのインスタンスを生成
use_pretrained = True # 学習済みのパラメータを使用
net = models.vgg16(pretrained=use_pretrained)
net.eval() # 推論モードに設定

class BaseTransform():
    """
    画像のサイズをリサイズし、色を標準化する。

    Attributes
    ---------------
    resize : int
        リサイズ先の画像の大きさ
    mean : (R, G, B)
        各色のチャンネルの平均値
    std : (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize), # 短い辺の長さがresizeの起き差になる
            transforms.CenterCrop(resize), # 画像中央を resize × resize で切り取り
            transforms.ToTensor(), # Torchテンソルに変換
            transforms.Normalize(mean, std) # 色情報の標準化
        ])

    # Pythonの一般的なメソッド。そのクラスのインスタンスが具体的な関数を指定されずに呼び出されたときに動作する関数
    def __call__(self, img):
        return self.base_transform(img)

# 画像前処理の動作を確認

# 画像の読み込み
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path) # (高さ、幅、色)

# 元の画像の表示
# plt.imshow(img)
# plt.show()

# 画像の前処理と処理済み画像の表示
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img) # size = (3, 224, 224)

# PytorchとPILでは画像の要素の順番が異なるため入れ替え
# Pytorch は[色チャネル、高さ、幅]
# PIL は[高さ、幅、色チャネル]
# img = Image.open(image_file_path) で取得した画像は、どっちでも扱えるみたいだが、Tensorで変換した後の要素の順番は、Pytorchのそれとなる
# img_transformed = img_transformed.numpy().transpose((1, 2, 0)) # size = (224, 224, 3)
# np.clip(img, 0, 1) = 0以下の値は0に、1以上の値は1にまとめる
# img_transformed = np.clip(img_transformed, 0, 1)
# plt.imshow(img_transformed)
# plt.show()

# 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor():
    """
    ILSVRCデータに対するもdるの出力からラベルを求める。

    Attributes
    -------------
    class_index : dictionary
        クラスindexとラベル名を対応させた辞書型変数
    """
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        確率が最大のILSVRCのラベル名を取得する
        Parameters
        ---------------
        out: torch.Size([1, 1000])
            Netからの出力

        Returns
        ---------------
        predicted_label_name : str
            最も予測確率が高いラベルの名前
        """
        maxid = np.argmax(out.detach().numpy()) # detach = requires_grad = Falseにする
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

# ILSVRCのラベル情報をロードし、辞書型変数を生成
ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))

predictor = ILSVRCPredictor(ILSVRC_class_index)

# 画像の読み込み
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path) # (高さ、幅、色)

# 前処理をして、バッチサイズの次元を追加する
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img) # size = (3, 224, 224)
inputs = img_transformed.unsqueeze_(0) # データをミニの形にする必要があるため４次元に変換 (ミニバッチ数 , 色チャネル, 高さ, 幅)

# モデルに入力し、モデル出力をラベルに変換
out = net(inputs) # size = (1, 1000)
result = predictor.predict_max(out)

# 予測結果の出力
print(result)