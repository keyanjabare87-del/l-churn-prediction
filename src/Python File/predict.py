
from pathlib import Path

import joblib
import pandas as pd


# 自动定位项目根目录
current_file = Path(__file__).resolve()
project_root = None

for parent in current_file.parents:
    if (parent / "models" / "random_forest_churn.joblib").exists():
        project_root = parent
        break

if project_root is None:
    raise FileNotFoundError(
        "未找到模型文件，请检查 models/random_forest_churn.joblib"
    )

# 加载完整 Pipeline
model_path = project_root / "models" / "random_forest_churn.joblib"
model = joblib.load(model_path)

# 当前项目在验证集上选出的阈值
threshold = 0.52

# 构造一条示例客户数据
# 字段与训练数据保持一致
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 660.0,
}])

# 预测流失概率
churn_probability = model.predict_proba(customer)[0, 1]

# 按验证集选出的阈值分类
prediction = int(churn_probability >= threshold)

print("========== 客户流失预测 ==========")
print(f"预测流失概率：{churn_probability:.2%}")
print(f"分类阈值：{threshold:.2f}")
print(f"预测结果：{'可能流失' if prediction == 1 else '可能不流失'}")
