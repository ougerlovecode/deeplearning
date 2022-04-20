import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
from resnet import *

import os

image_path = 'Dataset/image/'
train_set_path = 'Dataset/trainset.txt'
test_set_path = 'Dataset/testset.txt'
val_set_path = 'Dataset/validset.txt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据加载
class CifarDataset(Dataset):
    def __init__(self, mode):
        # 添加数据集的初始化内容
        self.mode = mode
        if self.mode == 'train':
            with open(train_set_path) as file_name:
                path_and_labeles = file_name.read().splitlines()
            self.pathes = []
            self.labels = []
            for path_and_label in path_and_labeles:
                path, label = path_and_label.split(' ')
                self.pathes.append(path)
                self.labels.append(label)
            self.len = len(self.pathes)
            assert self.len == 40000

        # elif self.mode == 'test':
        #     with open(test_set_path) as file_name:
        #         self.pathes = file_name.read().splitlines()
        #     self.len = len(self.pathes)
        #     assert self.len == 10000
        elif self.mode == 'val':
            with open(val_set_path) as file_name:
                path_and_labeles = file_name.read().splitlines()
            self.pathes = []
            self.labels = []
            for path_and_label in path_and_labeles:
                path, label = path_and_label.split(' ')
                self.pathes.append(path)
                self.labels.append(label)
            self.len = len(self.pathes)
            assert self.len == 10000

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        assert index < self.len
        path = image_path + self.pathes[index]
        if self.mode == 'train':
            img = Image.open(path)
            img = transform(img)
            return img, torch.tensor(int(self.labels[index]))
            # return img, torch.tensor(int(self.labels[index]))
        # elif self.mode == 'test':
        #     img = Image.open(path)
        #     img = transform(img)
        #     return img
        elif self.mode == 'val':
            img = Image.open(path)
            img = transform(img)
            return img, torch.tensor(int(self.labels[index]))

    def __len__(self):
        # 添加len函数的相关内容
        return self.len


# 构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 定义 train 函数
def train(epoch_num=30):
    # 参数设置

    val_num = 2
    running_loss = 0.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # Forward
            outputs = net(images)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)  # 计算单个batch误差
            # Backward
            loss.backward()
            #             # clip the grad
            #             clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            # Update
            optimizer.step()
            # 模型训练n轮之后进行验证
            # 打印log信息
            running_loss += loss.item()  # 200个batch的误差和
            if i % 20 == 19:  # 每200个batch打印一次训练状态
                print("[epoch %d,%5d/%d] loss: %.3f" \
                      % (epoch + 1, (i + 1) * batchsize, len(train_set), running_loss / 20))
                running_loss = 0.0

        if epoch % val_num == 1:
            validation()
            if not os.path.isdir('model'):
                os.mkdir('model')
            torch.save(net.state_dict(), "model/epoch" + str(epoch) + "model_params.pkl")
    print('Finished Training!')


# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in dev_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # 在这一部分撰写验证的内容
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("10000张测试集中的准确率为:%d %%" % (100 * correct / total))

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)


# 定义 test 函数
def test():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 将预测结果写入result.txt文件中，格式参照实验1
    net.eval()
    with open('test_result.txt', 'w') as f:
        f.write('')
    with open(test_set_path) as file_name:
        pathes = file_name.read().splitlines()
    # print(pathes)
    for index,path in enumerate(pathes):

        # imgs在前，labels在后，分别表示：图像转换0~1之间的值，labels为标签值。并且imgs和labels是按批次进行输入的。
        path = image_path + path
        image = Image.open(path)
        image = transform(image)
        image = image.cuda()
        # 计算图片在每个类别上的分数
        # print(image.shape)
        outputs = net(image.view(1,3,32,32))

        # 得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)  # torch.max()返回两个值，第一个值是具体的value，，也就是输出的最大值（我们用下划线_表示 ，指概率），
                                                # 第二个值是value所在的index（也就是predicted ， 指类别）
                                                # 选用下划线代表不需要用到的变量
                                                # 数字1：其实可以写为dim=1，表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
        with open('test_result.txt', 'a') as f:
            f.write(str(predicted.tolist()[0]))
            f.write('\n')

        if (index+1) %100 == 0:
            print("have test [{}/{}] image".format(index+1,len(pathes)))


batchsize = 128
if __name__ == "__main__":
    # 构建数据集
    train_set = CifarDataset('train')
    dev_set = CifarDataset('val')
    test_set = CifarDataset('test')

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batchsize, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=True)

    # 初始化模型对象
    # net = Net()
    net = ResNet18()
    #     net = VGG('VGG16')
    net = net.to(device)
    net.load_state_dict(torch.load("model/epoch17model_params.pkl"))
    # print(net)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,
                                 )

    # 模型训练
    # train(200)

    # 对模型进行测试，并生成预测结果
    test()
