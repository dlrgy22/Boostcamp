import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # conv층
        # 입력 체널, 출력 체널, 커널 사이즈,
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # fc층
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # forward가 정의되고 나면 backward 함수는 autograd를 사용하여 자동으로 정의
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)
    
    # CNN파라미터
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    my_input = torch.randn(1, 1, 32 ,32)
    output = net(my_input)
    target = torch.randn(10)
    target = target.view(1, -1)
    loss_function = nn.MSELoss()

    loss = loss_function(output, target)
    print(loss)

    net.zero_grad()
    print("conv1.bias.grad before backward")
    print(net.conv1.bias.grad)
    
    loss.backward()
    print("conv1.bias.grad after backward")
    print(net.conv1.bias.grad)

    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    optimizer.zero_grad()
    output = net(my_input)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
