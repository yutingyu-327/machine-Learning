from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. 数据加载与预处理
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# 编码标签为数值
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. GMM聚类分析
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)
probs = gmm.predict_proba(X_scaled)
clusters = gmm.predict(X_scaled)

# 分析误分样本
misclassified = np.where(clusters != y_encoded)[0]
print(f"\n误分样本数量: {len(misclassified)}")
print("\n误分样本特征统计:")
print(X.iloc[misclassified].describe())

# 3. 逻辑回归分类
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"\n逻辑回归测试集准确率: {lr.score(X_test, y_test):.2f}")

# 特征权重可视化
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Weight': lr.coef_[0]
}).sort_values('Weight', ascending=False)

sns.barplot(x='Weight', y='Feature', data=feature_importance)
plt.title('逻辑回归特征权重')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\n分析完成，结果已保存!")