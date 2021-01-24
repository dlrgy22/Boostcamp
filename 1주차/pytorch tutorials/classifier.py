import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CnnStructure(nn.Module):

    def __init__(self):
        super(CnnStructure, self).__init__()
        # conv층
        # 입력 체널, 출력 체널, 커널 사이즈,
        self.conv1 = nn.Conv2d(3, 6, 5)
        #maxpooling
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # fc층
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # xavier 초기화
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 출력단 활성화 함수 softmax
        return F.softmax(x, dim=1)


class Model():
    
    def __init__(self, train_dataset, test_dataset, classes):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes
        self.net = CnnStructure()


    def train(self):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        epoch = 2

        for e in range(epoch):
            running_loss = 0.0

            for i, data in enumerate(self.train_dataset, 0):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 2000 == 1999:
                    print(f"Train epoch : {e + 1}, {i + 1:5} loss: {running_loss / 2000 :.3}")
                    running_loss = 0.0

        print("Finished Training")


    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_dataset:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Acc : {100 * correct/total}")


    
def GetData():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    # 정답 레이블
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes



if __name__ == "__main__":

    train_set, test_set, classes = GetData()
    model = Model(train_dataset=train_set, test_dataset=test_set, classes=classes)
    model.train()
    model.test()






    
    # # data 다운 및 가져오기
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    # # 정답 레이블
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # cnn 사용
    # net = CNN_structure()
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    # for epoch in range(20):
    #     running_loss = 0.0

    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data

    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         if i % 2000 == 1999:
    #             print(f"Train epoch : {epoch + 1}, {i + 1:5}] loss: {running_loss / 2000 :.3}")
    #             running_loss = 0.0

    # print("Finished Training")

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    
    # print(f"Acc : {100 * correct/total}")
