#adaline 算法

import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, input_size, learning_rate=0.1, epochs=200):
        self.weights = np.zeros(input_size + 1)  # 初始化权重和偏置（包括偏置）
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 训练轮数

    def plot_line(self, w1, w2, b,color='k', x1_range=(-10, 10)):
        """
        根据权重 w1, w2 和偏置 b 绘制决策边界线。
        
        参数：
        - w1, w2, b: 权重和偏置
        - x1_range: x1 的取值范围，默认 (-10, 10)
        """
        print(w1,w2,b)
        # 计算 x1 范围内的 x2 值
        x1_values = np.linspace(x1_range[0], x1_range[1], 100)
        x2_values = - (w1 * x1_values + b) / w2  # 从方程推导得到 x2
        # 绘制线条
        plt.plot(x1_values, x2_values, label=f'Line: {w1}x1 + {w2}x2 + {b} = 0', color=color)

        
    def predict(self, X):
        # 预测函数：计算加权和并返回结果（线性激活）
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return z  # 返回加权和

    def train(self, X, y):
        # 设置图形
        plt.ion()  # 开启交互模式
        plt.figure(figsize=(8, 6))
        self.plot_decision_boundary(X, y)
        self.plot_line(1, -1, 1, 'r')
        
        # 训练 Adaline 模型
        # 记录训练过程中的损失
        losses = []
        for epoch in range(self.epochs):
            # 计算预测值
            predictions = self.predict(X)
            
            # 计算误差（真实值与预测值的差距）
            errors = y - predictions
            
            # 计算均方误差 (MSE)
            loss = (errors ** 2).mean()
            losses.append(loss)
            
            # 更新权重和偏置（梯度下降法）
            self.weights[1:] += self.learning_rate * np.dot(X.T, errors) / len(X)
            self.weights[0] += self.learning_rate * errors.sum() / len(X)
                
            # 每一轮训练后，绘制决策边界
            # 根据权重画出来决策边界线
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")
            self.plot_line(self.weights[1], self.weights[2], self.weights[0]+1)
            plt.pause(0.1)  # 暂停，让图形更新
        
        plt.ioff()  # 关闭交互模式
        plt.show()  # 显示最终图形
        
    def plot_decision_boundary(self, X, y, ):
        n_samples = len(X)  # 计算样本数量
        # 画出训练数据点
        # 前面一半用蓝点
        plt.scatter(X[:n_samples // 2, 0], X[:n_samples // 2, 1], c='b', cmap=plt.cm.Paired, marker='o', s=1, edgecolor='b')
        # 后面一半用黄点
        plt.scatter(X[n_samples // 2:, 0], X[n_samples // 2:, 1], c='y', cmap=plt.cm.Paired, marker='x', s=1, edgecolor='y')
        
        
        # 设置图形的标签
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper right')

# 训练数据
# 设置随机种子，确保结果可复现
np.random.seed(42)

# 生成x值
n_samples = 50  # 每类样本数量
x = np.random.uniform(-5, 5, n_samples)

# 生成 y = x + 1 对应的理想值
y_true = x + 1

# 第一类数据 (标签 -1)：位于 y = x + 1 下方
noise_class_1 = np.random.uniform(-5, 0, n_samples)  # 随机噪声使得y值小于直线
y_class_1 = y_true + noise_class_1  # 类别1的y值

# 第二类数据 (标签 +1)：位于 y = x + 1 上方
noise_class_2 = np.random.uniform(0, 5, n_samples)  # 随机噪声使得y值大于直线
y_class_2 = y_true + noise_class_2  # 类别2的y值

# 合并数据
X = np.vstack([np.column_stack((x, y_class_1)), np.column_stack((x, y_class_2))])
y = X[:, 0] - X[:, 1] + 1 
print(X)
print(y)
#z = x - y + 1


# 创建 Adaline 实例并训练
adaline = Adaline(input_size=2, epochs=50, learning_rate=0.01)
adaline.train(X, y)

