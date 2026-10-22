from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

current_file = Path(__file__).resolve()
Root = next(
    parent
    for parent in current_file.parents
    if (parent/"data"/"raw").is_dir()
)

DATA_PATH = (
    Root / "data" / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# 清洗TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(),
                                   errors="coerce").fillna(0)

print("数据读取成功！")
print("数据规模：", df.shape)

# 分离x和y

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({
    "No": 0, "Yes": 1
})

if y.isna().any():
    raise ValueError("Churn 中存在未预期的标签")

print("特征数量：", X.shape[1])
print("流失客户比例：", round(y.mean() * 100, 2), "%")

# 第一次划分：80% 开发数据，20% 最终测试数据
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 第二次划分：开发数据中 75% 用于训练，25% 用于验证
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print("训练集规模：", X_train.shape)
print("验证集规模：", X_val.shape)
print("测试集规模：", X_test.shape)

# 自动识别数值列和类别列

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category","bool"]
).columns.tolist()

print("数值特征：", numeric_features)
print("类别特征数量：", len(categorical_features))

# 数值特征：填补缺失值，再标准化
numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
])

# 类别特征：填补缺失值，再独热编码
categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# 建立Baseline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=42
    )),
])

model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:,1]

print("\n========== 基线模型评估 ==========")
print(f"Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, y_prob):.4f}")

print("\n分类报告：")
print(classification_report(
    y_val,
    y_pred,
    target_names=["未流失", "已流失"],
    zero_division=0,
))

print("混淆矩阵（行是真实标签，列是预测标签）：")
print(confusion_matrix(y_val, y_pred))

print("\n基线模型训练与评估完成！")

# =========================
# 7. 训练随机森林模型
# =========================

print("\n========== 随机森林模型训练 ==========")

rf_model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
# 只使用训练集训练
rf_model.fit(X_train, y_train)

# 在验证集上评估
y_rf_pred = rf_model.predict(X_val)
y_rf_prob = rf_model.predict_proba(X_val)[:, 1]

print("\n========== 随机森林验证集评估 ==========")

print(
    f"Accuracy : "
    f"{accuracy_score(y_val, y_rf_pred):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_val, y_rf_pred, zero_division=0):.4f}"
)
print(
    f"Recall   : "
    f"{recall_score(y_val, y_rf_pred, zero_division=0):.4f}"
)
print(
    f"F1       : "
    f"{f1_score(y_val, y_rf_pred, zero_division=0):.4f}"
)
print(
    f"ROC-AUC  : "
    f"{roc_auc_score(y_val, y_rf_prob):.4f}"
)

print("\n随机森林分类报告：")
print(classification_report(
    y_val,
    y_rf_pred,
    target_names=["未流失", "已流失"],
    zero_division=0,
))

print("随机森林混淆矩阵：")
print(confusion_matrix(y_val, y_rf_pred))

# =========================
# 随机森林参数对比实验
# =========================

rf_configs = [
    {"max_depth": 8, "min_samples_leaf": 2},
    {"max_depth": 8, "min_samples_leaf": 5},
    {"max_depth": 12, "min_samples_leaf": 2},
    {"max_depth": 12, "min_samples_leaf": 5},
]

rf_results = []
best_rf_model = None
best_rf_f1 = -1

for config in rf_configs:
    print(f"\n正在训练参数组合：{config}")

    candidate = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    candidate.fit(X_train, y_train)

    # 用验证集评估默认阈值 0.5
    val_pred = candidate.predict(X_val)
    val_prob = candidate.predict_proba(X_val)[:, 1]

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )
    auc = roc_auc_score(y_val, val_prob)

    result = {
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }
    rf_results.append(result)

    print(
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}, "
        f"ROC-AUC={auc:.4f}"
    )

    if f1 > best_rf_f1:
        best_rf_f1 = f1
        best_rf_model = candidate

# 保存本轮表现最好的模型，供后续评估使用
rf_model = best_rf_model
y_rf_pred = rf_model.predict(X_val)
y_rf_prob = rf_model.predict_proba(X_val)[:, 1]

print("\n========== 随机森林参数实验汇总 ==========")
for result in sorted(
    rf_results,
    key=lambda item: item["f1"],
    reverse=True,
):
    print(result)

print(f"\n本轮最高验证集 F1：{best_rf_f1:.4f}")

# =========================
# 随机森林：验证集阈值实验
# =========================

print("\n========== 随机森林验证集阈值对比 ==========")

threshold_results = []

for threshold in np.arange(0.20, 0.81, 0.01):
    y_threshold_pred = (y_rf_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, y_threshold_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_threshold_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_threshold_pred, zero_division=0
    )

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

threshold_df = pd.DataFrame(threshold_results)

# 按 F1 从高到低排序
threshold_df = threshold_df.sort_values(
    by="f1", ascending=False
)

print("\n========== 验证集最佳阈值搜索结果 ==========")
print(threshold_df.head(10).to_string(index=False))

best_threshold = threshold_df.iloc[0]["threshold"]

print(f"\n当前搜索范围内最佳阈值：{best_threshold:.2f}")
print(f"对应 Precision：{threshold_df.iloc[0]['precision']:.4f}")
print(f"对应 Recall：{threshold_df.iloc[0]['recall']:.4f}")
print(f"对应 F1：{threshold_df.iloc[0]['f1']:.4f}")

# ========== 特征重要性分析：Permutation Importance ==========

print("\n========== 特征重要性分析 ==========")

importancce_results = permutation_importance(
    estimator=rf_model,
    X=X_val,
    y=y_val,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": importancce_results.importances_mean,
    "importance_std": importancce_results.importances_std,
})

importance_df = importance_df.sort_values(
    by="importance_mean", ascending=False
)

print("\n特征重要性 Top 15：")
print(
    importance_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x: 5f}",
    )
)# 保存特征重要性结果
report_dir = Root / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

importance_path = report_dir / "feature_importance.csv"
importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"\n特征重要性结果已保存：{importance_path}")


# ========== 最终测试集评估 ==========

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# 阈值沿用验证集选出的结果
final_threshold = 0.52

# 在测试集上预测
y_test_prob = rf_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= final_threshold).astype(int)

print("\n========== 最终测试集评估 ==========")
print(f"分类阈值：{final_threshold:.2f}")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_test_prob):.4f}")

print("\n测试集分类报告：")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["未流失", "已流失"],
        zero_division=0,
    )
)

print("\n测试集混淆矩阵（行是真实标签，列是预测标签）：")
print(confusion_matrix(y_test, y_test_pred))


# ========== 保存最终模型 ==========

import joblib
import json

# 创建模型保存目录
model_dir = Root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# 保存完整 Pipeline
model_path = model_dir / "random_forest_churn.joblib"
joblib.dump(rf_model, model_path)

# 保存本次最终测试集指标
test_metrics = {
    "model": "RandomForestClassifier",
    "max_depth": 8,
    "min_samples_leaf": 2,
    "threshold": 0.52,
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "precision": float(
        precision_score(y_test, y_test_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_test_pred, zero_division=0)
    ),
    "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
}

metrics_path = report_dir / "test_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, ensure_ascii=False, indent=4)

print(f"\n模型已保存：{model_path}")
print(f"测试集指标已保存：{metrics_path}")
