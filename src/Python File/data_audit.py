from multiprocessing.reduction import duplicate
from pathlib import Path
import pandas as pd

# 获取项目根目录
ROOT = Path(__file__).resolve().parents[2]

# 拼接csv文件路径
DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# 检查文件是否存在
if not DATA_PATH.exists():
    raise FileExistsError(
        f"找不到数据文件：{DATA_PATH}"
    )

# 读取CSV
df = pd.read_csv(DATA_PATH)

# 输出数据规模
print("数据读取成功")
print("数据行数： ", df.shape[0])
print("数据列数： ", df.shape[1])

print("\n数据前五行： ")
print(df.head())

# 检查每一列的数据类型
print("\\\n各列数据类型: ")
print(df.dtypes)

# 检查每一列的缺失值数量
print("\n各列缺失值数量：")
print(df.isnull().sum())

# 检查重复行
duplicate_count = df.duplicated().sum()
print("\n重复行数量：", duplicate_count)

# 检查数据基本统计信息
print("\n数据基本统计信息：")
print(df.describe())

# 检查TotalCharges中是否存在空字符串
total_charges_text = df["TotalCharges"].astype(str).str.strip()

blank_count = (total_charges_text == "").sum()

print("\nTotalCharges空字符串数量： ", blank_count)

# 尝试讲TotalCharges转换为数值
total_charges_numeric = pd.to_numeric(total_charges_text, errors="coerce").fillna(0).astype(float)

print("\nTotalCharges 修复后类型：", df["TotalCharges"].dtype)
print("TotalCharges 修复后缺失值：", df["TotalCharges"].isna().sum())

# 检查目标变量Churn的分布
print("\n客户流失情况：")
print(df["Churn"].value_counts())

print("\n客户流失比例：")
print(
    (df["Churn"].value_counts(normalize=True)*100)
    .round(2)
    .astype(str)
    .add("%")
)

# 计算流失值
churn_rate = (df["Churn"] == "Yes").mean()*100
print(f"\n总体客户流失率: {churn_rate:.2f}%")

# 检查无法转化为数字的记录
invalid_count = total_charges_numeric.isna().sum()

print("TotalCharges 无法转换为数值的记录数：", invalid_count)

# 查看异常记录
if invalid_count > 0:
    print("\n异常记录示例：")
    print(
        df.loc[
            total_charges_numeric.isna(),
            ["customerID", "tenure", "TotalCharges","Churn"],
        ].head(10)
    )