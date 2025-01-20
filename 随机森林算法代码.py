from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

categorical_cols = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 
                    'Season', 'Subscription Status', 'Payment Method', 'Shipping Type', 
                    'Promo Code Used', 'Preferred Payment Method', 'Frequency of Purchases']

'''
LabelEncoder 是 scikit-learn 库（sklearn.preprocessing 模块）中的一个类，
主要用于将分类标签转换为数字编码。这在机器学习任务中非常有用，
因为许多机器学习算法要求输入数据为数值型，而不是字符串或其他类型的类别标签。
'''
encoder = LabelEncoder()

for col in  categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Features (X) and Label (y)
X = data.drop(columns=['Customer ID', 'Subscription Status'])  # 将ID与label给去掉
y = data['Subscription Status']  # label

numerical_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
'''
从 scikit-learn 的预处理模块中导入 StandardScaler 类。
StandardScaler 是一个常用的工具，用于对数据进行标准化处理，
将数据转换为均值为 0，标准差为 1 的标准正态分布。
'''
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# 将标签编码应用于剩余的对象类型列
# tpye 类型为'object'的则是未处理成标签编码的列类型
for col in X.select_dtypes(include='object').columns:
    X[col] = encoder.fit_transform(X[col])

#划分为 train and test 数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 

data['Discount Applied']= label_encoder.fit_transform(data['Discount Applied'])

model_RF = RandomForestClassifier(random_state=42, n_estimators=100)
model_RF.fit(X_train, y_train)

y_pred = model_RF.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#重要性排序
importances = model_RF.feature_importances_
features = X.columns

# 将特征重要性和特征名称组合在一起，并按照重要性进行降序排序
feature_importance_data = sorted(zip(importances, features), reverse=True)
importances_sorted, features_sorted = zip(*feature_importance_data)

plt.figure(figsize=(12, 8))
# 绘制柱状图，按照降序排列的顺序绘制
sns.barplot(x=importances_sorted, y=features_sorted, palette='viridis')

# 计算重要性的均值
avg_importance = np.mean(importances_sorted)

# 添加红色竖立的虚线表示重要性均值
plt.axvline(x=avg_importance, color='r', linestyle='--', label=f'avg_importance={avg_importance:.2f}')

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
# 添加图例，设置图例位置等属性让其显示更合理
plt.legend(fontsize='medium')
plt.show()