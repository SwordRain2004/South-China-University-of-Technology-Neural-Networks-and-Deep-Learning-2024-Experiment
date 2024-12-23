import numpy as np
# 三、多层感知机实现异或运算（提交实现步骤描述、源代码以及最后的测试误差）  要求：不允许使用 Pytorch/Tensorflow 等深度学习框架，使用 Python 实现网络的前向传播和反向传播过程。
# 数据集：
# [[[0, 0], [0]],
# [[0, 1], [1]],
# [[1, 0], [1]],
# [[1, 1], [0]]]
# BP 算法实现
# 可供参考的实现过程： 单个神经元操作：
# 1、定义参数矩阵、定义激活函数、定义损失函数
# 2、计算神经元输出
# 3、计算误差
# 实现前向传播：
# 1、定义神经元个数
# 2、计算层中每个神经元的输出
# 实现反向传播：
# 1、计算误差
# 2、计算每一层中权重的梯度
# 3、更新输出层权重
# 4、更新隐含层权重
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(0)
input_size = 2
hidden_size = 2
output_size = 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y = sigmoid(Z2)
    return Y, A1
def backward(X, y, Y, A1, W2):
    m = y.shape[0]
    dY = Y - y
    dW2 = (1 / m) * np.dot(A1.T, dY)
    db2 = (1 / m) * np.sum(dY, axis=0, keepdims=True)
    dA1 = np.dot(dY, W2.T) * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(X.T, dA1)
    db1 = (1 / m) * np.sum(dA1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2
learning_rate = 0.1
num_epochs = 10000
for epoch in range(num_epochs):
    Y, A1 = forward(X, W1, b1, W2, b2)
    loss = mse_loss(y, Y)
    dW1, db1, dW2, db2 = backward(X, y, Y, A1, W2)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1.flatten()
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2.flatten()
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
Y_pred, _ = forward(X, W1, b1, W2, b2)
print("Predicted outputs:")
print(Y_pred)
final_loss = mse_loss(y, Y_pred)
print(f'Final Test Loss: {final_loss:.4f}')