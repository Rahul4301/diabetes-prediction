import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1️⃣ 載入已保存的模型
loaded_model = joblib.load('xgboost_diabetes_model.pkl')
print("✅ 模型已成功載入！")

# 2️⃣ 讀取數據
data = pd.read_csv('diabetes.csv')

# 3️⃣ 定義特徵與目標變數
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 4️⃣ 替換 0 為 NaN
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[cols_to_replace] = X[cols_to_replace].replace(0, pd.NA)

# 5️⃣ 填補 NaN（使用中位數）
X.fillna(X.median(), inplace=True)

# 6️⃣ 確保 `random_state=42`，保持與訓練時相同的數據切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7️⃣ 標準化特徵（必須與訓練時一致）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # **載入後的模型也需要標準化數據！**

# 8️⃣ 使用載入的模型進行預測
y_prob_xgb = loaded_model.predict_proba(X_test)[:, 1]  # 獲取預測機率
threshold = 0.6
y_pred_xgb = (y_prob_xgb > threshold).astype(int)

# 9️⃣ **測試載入的模型 Performance**
print("\n📊 Performance of Loaded Model (Threshold = 0.6):")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))

# 🔟 **檢查是否與原始模型一致**
print("\n🔍 Classification Report of Loaded Model:")
print(classification_report(y_test, y_pred_xgb))

# 1️⃣1️⃣ **視覺化混淆矩陣**
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Loaded Model, Threshold = 0.6)")
plt.show()
