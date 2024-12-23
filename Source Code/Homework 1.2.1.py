import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# (1) 生成训练数据，数据为带有服从-0.5 到 0.5 的均匀分布噪声的正弦函数，代码如下：
num_observations = 100
x = np.linspace(-3,3,num_observations)
y = np.sin(x) + np.random.uniform(-0.5,0.5,num_observations)
# 画出这 100 个样本的散点图。（提交散点图）
plt.scatter(x, y, color='blue', label='Data with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Noisy Sine Function')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('homework_1_2_1_1.png')
# (2)使用 Pytorch实现线性回归模型y=w*x+b，训练参数w和b。
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
# (3) 输出参数w、b和损失。（提交运行结果）
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f'w: {w}, b: {b}, Loss: {loss.item():.4f}')
# (4) 画出预测回归曲线以及训练数据散点图，对比回归曲线和散点图并分析原因。
model.eval()
with torch.no_grad():
    y_pred = model(x_train).numpy()
plt.scatter(x, y, color='blue', label='Data with Noise')
plt.plot(x, y_pred, color='red', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression on Noisy Sine Function')
plt.legend()
plt.grid(True)
# plt.savefig('homework_1_2_1_4.png')
plt.show()