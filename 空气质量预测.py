import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 设置中文字体 - 使用系统支持的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载
data = pd.read_excel(r"D:\202330906135张雨桐\空气质量预测代码\空气质量数据\AirQualityUCI.xlsx")
print(f"数据列名: {data.columns.tolist()}")

# 2. 数据预处理
# 处理日期时间列
data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
# 提取时间特征
data['Hour'] = data['Datetime'].dt.hour
data['DayOfWeek'] = data['Datetime'].dt.dayofweek
data['Month'] = data['Datetime'].dt.month
data['Day'] = data['Datetime'].dt.day
data['Year'] = data['Datetime'].dt.year
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

# 异常值处理
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
time_feature_cols = ['Hour', 'DayOfWeek', 'Month', 'Day', 'Year', 'IsWeekend']
original_numeric_cols = [col for col in numeric_cols if col not in time_feature_cols]

def handle_outliers(df, column):
    # 计算统计指标
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # 定义异常值边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别异常值
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    if outliers_mask.any():
        # 用中位数替换异常值
        median_val = df[column].median()
        df.loc[outliers_mask, column] = median_val
        return outliers_mask.sum()
    return 0

# 应用异常值处理
total_outliers = 0
for col in original_numeric_cols:
    outliers_count = handle_outliers(data, col)
    total_outliers += outliers_count

print(f"\n总共处理了 {total_outliers} 个异常值")

# 选择特征和目标变量
target_column = 'CO(GT)'

# 排除不需要的列
exclude_cols = ['Date', 'Time', 'Datetime', target_column]
feature_cols = [col for col in data.columns if col not in exclude_cols]

X = data[feature_cols]
y = data[target_column]

# 检查缺失值并处理
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    print(f"发现缺失值: {missing_values[missing_values > 0].to_dict()}")
    X = X.fillna(X.median())
    print("已使用中位数填充缺失值")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 模型训练
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 模型评估
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型性能评估:")
print(f"R²分数: {r2:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")

# 计算预测准确率
accuracy_threshold = 0.1  # 10%误差范围
accurate_predictions = np.sum(np.abs((y_test - y_pred) / y_test) <= accuracy_threshold)
accuracy_rate = accurate_predictions / len(y_test) * 100
print(f"预测准确率(误差≤10%): {accuracy_rate:.2f}%")

# 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n前10个最重要的特征:")
for i, row in feature_importance.head(10).iterrows():
    print(f"{row['特征']}: {row['重要性']:.4f}")

# 可视化特征重要性
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance['特征'][:15], feature_importance['重要性'][:15])
plt.xlabel('特征重要性', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.title('空气质量预测 - 特征重要性排名(前15个)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 在柱状图上添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{width:.4f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('特征重要性.png', dpi=300, bbox_inches='tight')
print("\n特征重要性图已保存为 '特征重要性.png'")

# 可视化预测结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 实际值与预测值的散点图：
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际值', fontsize=11)
axes[0, 0].set_ylabel('预测值', fontsize=11)
axes[0, 0].set_title('实际值与预测值的散点图', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 残差图
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('预测值', fontsize=11)
axes[0, 1].set_ylabel('残差', fontsize=11)
axes[0, 1].set_title('残差分布图', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 预测对比折线图
sample_indices = np.arange(min(50, len(y_test)))
axes[1, 0].plot(sample_indices, y_test.values[:50], label='实际值',
                marker='o', markersize=4, linewidth=2, alpha=0.8)
axes[1, 0].plot(sample_indices, y_pred[:50], label='预测值',
                marker='s', markersize=4, linewidth=2, alpha=0.8, linestyle='--')
axes[1, 0].set_xlabel('样本序号', fontsize=11)
axes[1, 0].set_ylabel('CO(GT)浓度', fontsize=11)
axes[1, 0].set_title('前50个样本预测对比', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 误差分布直方图
error_percentage = np.abs((y_test - y_pred) / y_test) * 100
axes[1, 1].hist(error_percentage, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].axvline(x=10, color='red', linestyle='--', linewidth=2, label='10%误差线')
axes[1, 1].set_xlabel('预测误差百分比 (%)', fontsize=11)
axes[1, 1].set_ylabel('样本数量', fontsize=11)
axes[1, 1].set_title('预测误差分布', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('空气质量预测模型结果分析', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('预测结果分析.png', dpi=300, bbox_inches='tight')
print("预测结果分析图已保存为 '预测结果分析.png'")

# 保存结果（移除了模型文件保存）
feature_importance.to_csv('特征重要性.csv', index=False, encoding='utf-8-sig')

results_df = pd.DataFrame({
    '实际值': y_test.values,
    '预测值': y_pred,
    '误差': y_test.values - y_pred,
    '绝对误差': np.abs(y_test.values - y_pred),
    '误差百分比': np.abs((y_test.values - y_pred) / y_test.values) * 100
})
results_df.to_csv('预测结果.csv', index=False, encoding='utf-8-sig')

print("特征重要性: 特征重要性.csv")
print("预测结果: 预测结果.csv")

print(f"\n模型训练完成!")
print(f"最终R²分数: {r2:.4f}")
print(f"预测准确率(误差≤10%): {accuracy_rate:.2f}%")

# 显示图表
plt.show()

# 查看目标变量的分布
print(f"\n目标变量统计:")
print(f"  最小值: {y.min():.2f}")
print(f"  最大值: {y.max():.2f}")
print(f"  平均值: {y.mean():.2f}")
print(f"  标准差: {y.std():.2f}")
print(f"  变异系数(CV): {y.std()/y.mean()*100:.2f}%")

# 检查异常值
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
outliers_ratio = ((y < (Q1 - 1.5*IQR)) | (y > (Q3 + 1.5*IQR))).sum() / len(y) * 100
print(f"  异常值比例: {outliers_ratio:.2f}%")