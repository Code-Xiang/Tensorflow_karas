import paddle
import paddle
from paddle import nn
from paddle.vision import datasets
from paddle.vision import transforms
import paddle.nn.functional as F

# 获取数据集
transform = transforms.Compose([transforms.Normalize([127.5], [127.5])])
train_dataset = datasets.MNIST(mode='train', transform=transform)
test_dataset = datasets.MNIST(mode='test', transform=transform)

# 设置网络模型的函数
class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

net = LeNet()
model = paddle.Model(net)

# 创建学习策略
loss = nn.CrossEntropyLoss()

# 创建优化算法
trainer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

# 加载数据集
train_dataLoader = paddle.io.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# 手动写训练代码
num_epochs = 5
for epoch in range(num_epochs):
    print("第{}次训练开始了".format(epoch+1))
    total_loss = 0
    for batch_id, (x, y) in enumerate(train_dataLoader):
        output = net(x)
        running_loss = loss(output, y)
        running_loss.backward()
        trainer.step()
        trainer.clear_grad()
        acc = paddle.metric.accuracy(output, y)
        if(batch_id % 100 == 0):
            print("第{}次训练，损失为：{}, 精确率为:{}".format(batch_id, running_loss, acc))

