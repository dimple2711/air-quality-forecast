import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体 - 使用系统支持的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 简单的梯度下降线性回归
class SimpleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    # 训练模型
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算损失
            loss = np.mean((y_pred - y) ** 2) / 2
            self.loss_history.append(loss)

            # 计算梯度
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # 预测
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 简单的IQR异常值处理
def simple_outlier_handling(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # 计算边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 将异常值替换为边界值
    data[column] = np.clip(data[column], lower_bound, upper_bound)
    return data

# 评估模型
def evaluate_model(y_true, y_pred):
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # 准确率
    errors = np.abs((y_true - y_pred) / y_true)
    accuracy = np.mean(errors <= 0.1) * 100

    return r2, rmse, mae, accuracy

def main():
    # 加载数据
    df = pd.read_excel(r"D:\202330906135张雨桐\data\AirQualityUCI.xlsx")

    print(f"原始数据形状: {df.shape}")

    # 简单的异常值处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df = simple_outlier_handling(df, col)

    # 特征选择
    features = ['PT08.S2(NMHC)', 'NOx(GT)', 'C6H6(GT)', 'NO2(GT)',
                'PT08.S1(CO)', 'T', 'PT08.S5(O3)', 'PT08.S4(NO2)',
                'PT08.S3(NOx)']

    # 检查哪些特征存在
    features = [f for f in features if f in df.columns]
    print(f"使用的特征: {features}")

    # 目标变量
    target = 'CO(GT)'

    # 准备数据
    X = df[features].values
    y = df[target].values

    # 标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # 训练测试分割
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 训练模型
    model = SimpleLinearRegressionGD(learning_rate=0.05, n_iterations=1500)
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred = model.predict(X_test)
    r2, rmse, mae, accuracy = evaluate_model(y_test, y_pred)

    print(f"\nR²分数: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"准确率(误差≤10%): {accuracy:.2f}%")

    # 特征重要性
    feature_importance = np.abs(model.weights)
    feature_importance = feature_importance / np.sum(feature_importance)

    importance_df = pd.DataFrame({
        '特征': features,
        '重要性': feature_importance
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性分析")
    print(importance_df.to_string(index=False))

    # 可视化
    plt.figure(figsize=(15, 4))

    # 损失曲线
    plt.subplot(131)
    plt.plot(model.loss_history)
    plt.title('损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')

    # 预测值与真实值的对比
    plt.subplot(132)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值与真实值的对比')

    # 特征重要性
    plt.subplot(133)
    plt.barh(range(len(features)), importance_df['重要性'].values)
    plt.yticks(range(len(features)), importance_df['特征'].values)
    plt.xlabel('重要性')
    plt.title('特征重要性')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig('空气质量预测结果.png', dpi=300)
    plt.show()

    # 保存结果
    importance_df.to_csv('特征重要性.csv', index=False, encoding='utf-8-sig')

    results_df = pd.DataFrame({
        '真实值': y_test,
        '预测值': y_pred,
        '误差': y_test - y_pred
    })
    results_df.to_csv('预测结果.csv', index=False, encoding='utf-8-sig')

    print("\n结果已保存：")
    print("空气质量预测结果.png")
    print("特征重要性.csv")
    print("预测结果.csv")

    return model, r2, rmse, mae, accuracy


if __name__ == "__main__":
    model, r2, rmse, mae, accuracy = main()