import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

test_data_path = "./datas/dataset/test/images/"
train_data_path = "./datas/dataset/train/images/"
train_label_path = "./datas/dataset/train/labels_train.txt"
val_data_path = "./datas/dataset/val/images/"
val_label_path = "./datas/dataset/val/labels_val.txt"
batchsize = 64
learning_rate = 1e-2

# 参考 https://blog.csdn.net/xjm850552586/article/details/109171016
class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self, mode):
        # 这里添加数据集的初始化内容
        self.mode = mode
        if mode == "test":
            self.root_path = test_data_path
            self.image_paths = os.listdir(test_data_path)
        if mode == "train":
            with open(train_label_path) as file_name:
                numbers = file_name.read().splitlines()
            self.root_path = train_data_path
            self.image_paths = os.listdir(train_data_path)
            self.label = torch.tensor([int(i) for i in numbers])
        if mode == "val":
            with open(val_label_path) as file_name:
                numbers = file_name.read().splitlines()
            self.root_path = val_data_path
            self.image_paths = os.listdir(val_data_path)
            self.label = torch.tensor([int(i) for i in numbers])

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        ])
        assert index < len(self.image_paths)
        if self.mode == "test":
            path = os.path.join(self.root_path ,'test_' +str(index) +'.bmp')
            img = Image.open(path)
            img = transform(img)
            return img
        elif self.mode == 'train':
            path = os.path.join(self.root_path , 'train_' + str(index) + '.bmp')
            img = Image.open(path)
            img = transform(img)
            return img, self.label[index]
        else:
            path = os.path.join(self.root_path , 'val_' + str(index) + '.bmp')
            img = Image.open(path)
            img = transform(img)
            return img, self.label[index]

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.image_paths)


class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        self.layer1 = nn.Sequential(
            nn.Linear(28*28,300),
            nn.BatchNorm1d(300),nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(300,100),
            nn.BatchNorm1d(100),nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(100,10)
        )

    def forward(self, x):
        # 此处添加模型前馈函数的内容，return函数需自行修改
        x = x.view(-1, 28 * 28 * 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def validation(model,loader):
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    eval_loss = 0
    eval_acc = 0
    model.eval()
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in loader:
            # 在这一部分撰写验证的内容，下面两行不必保留
            img, label = data
            output = model(img)
            loss = loss_function(output, label)
            eval_loss += loss.data * img.size(0)
            _, pred = torch.max(output, 1)
            num_correct = (pred == label).sum().item()
            eval_acc += num_correct
    print("预测正确的数量", eval_acc)
    print("总数量", len(loader.dataset))
    print("Test Loss:{:.6f},Acc:{:.6f}".
          format(eval_loss / len(loader.dataset), eval_acc / len(loader.dataset)))


def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def alltest(model , loader):
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    ])
    model.eval()
    with open('test_result.txt', 'w') as f:
        f.write('')
    for index in range(len(dataset_test)):
        img = Image.open(dataset_test.root_path + 'test_' + str(index) + '.bmp')
        img = transform(img)
        output = model(img)
        _, pred = torch.max(output, 1)
        with open('test_result.txt', 'a') as f:
            f.write(str(pred.tolist()[0]))
            f.write('\n')
    # 将结果按顺序写入txt文件中，下面一行不必保留
def train(epoch_num):
    # 循环外可以自行添加必要内容
    for index, data in enumerate(data_loader_train):
        images, true_labels = data
        # print(true_labels[0])
        # print(images[0])
        # imshow(images[0])
        # assert 0
        # 该部分添加训练的主要内容
        output = model(images)
        print(output)
        print(true_labels)
        assert 0
        loss = loss_function(output, true_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 必要的时候可以添加损失函数值的信息，即训练到现在的平均损失或最后一次的损失，下面两行不必保留
        if index % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, index * len(images), len(dataset_train),
                       100. * index / len(data_loader_train), loss.item()))
    if epoch_num % 10 == 0:
        torch.save(model.state_dict(),"model/epoch"+ str(epoch_num) + "model_params.pkl")

if __name__ == "__main__":

    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset(mode='train')

    dataset_val = HandWritingNumberRecognize_Dataset(mode='val')

    dataset_test = HandWritingNumberRecognize_Dataset(mode='test')

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train ,batch_size=batchsize, shuffle=True,num_workers=2)

    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batchsize, shuffle=True,num_workers=2)

    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batchsize, shuffle=True,num_workers=2)

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network()

    # 损失函数设置
    loss_function =torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)  # torch.optim中的优化器进行挑选，并进行参数设置

    max_epoch = 51  # 自行设置训练轮数
    num_val = 10  # 经过多少轮进行验证


    # 然后开始进行训练
    for epoch in range(max_epoch):
        train(epoch)
        # 在训练数轮之后开始进行验证评估
        if epoch % num_val == 0:
            validation(model, data_loader_val)

    # model.load_state_dict(torch.load("model2/epoch10model_params.pkl"))
    #
    # # 自行完善测试函数，并通过该函数生成测试结果
    # alltest(model, data_loader_test)
