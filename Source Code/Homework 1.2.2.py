import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# 2、线性回归（使用多项式函数对原始数据进行变换）
# (1) 生成训练数据，数据同上
num_observations = 100
x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
# (2) 使用 Pytorch/Tensorflow 实现线性回归模型，这里我们假设y是x的 3 次多项式，那么我们可以将数据扩展为：x、x2、x3三维数据，此时模型变为：
# 𝑦 = 𝑤1 ∗ 𝑥 + 𝑤2 ∗ 𝑥2 + 𝑤3 ∗ 𝑥3 + 𝑏
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)
x_train_poly = torch.cat([x_train, x_train**2, x_train**3], dim=1)
# (3) 训练模型并输出参数w1、w2、w3、b和损失。（提交运行结果）
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)
model = PolynomialRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_poly)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
w1, w2, w3 = model.linear.weight[0].detach().numpy()
b = model.linear.bias.item()
print(f'w1: {w1}, w2: {w2}, w3: {w3}, b: {b}, Loss: {loss.item():.4f}')
# (4) 画出预测回归曲线以及训练数据散点图，对比回归曲线和散点图并分析原因。
model.eval()
with torch.no_grad():
    y_pred = model(x_train_poly).numpy()
plt.scatter(x, y, color='blue', label='Data with Noise')
plt.plot(x, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression on Noisy Sine Function')
plt.legend()
plt.grid(True)
# plt.savefig('homework_1_2_2_4.png')
plt.show()