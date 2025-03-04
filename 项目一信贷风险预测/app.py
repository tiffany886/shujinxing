# %% [markdown]
# # 基于机器学习可解释性分析的信贷预测案例研究

# %%
# !pip install imbalanced-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install shap -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple

# %%
# 忽视警告，
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import streamlit as st

# 使用pandas的read_csv函数读取CSV文件
data = pd.read_csv("C:\\Users\\Tiffany\\Desktop\\python\\shujinxing\\项目一信贷风险预测\\data.csv")

# # 随机打乱数据集，这一步的操作是可选的，类似于把练习题不断更改顺序，防止电脑学习到固定的顺序，这一步是可选的
# random_seed = 77
# data = data.sample(frac=1, random_state=random_seed)

# 显示前几行数据，以确认数据已正确加载
st.subheader("原始数据前几行")
st.write(data.head())

# 显然id列是噪声列，可以删除
data = data.drop(columns=['Id'])
st.subheader("删除Id列后的数据前几行")
st.write(data.head())

st.subheader("数据集的行数和列数")
st.write(data.shape)

st.subheader("数据集信息")
st.write(data.info())

st.subheader("数据集描述统计信息")
st.write(data.describe())

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # preprocessing预处理模块 - 用于特征标准化、独热编码和数值编码
from sklearn.model_selection import train_test_split  # model_selection 模型选择 - 用于将数据集划分为训练集和测试集

# 分离特征数据和标签数据
X = data.drop(['Credit Default'], axis=1)  # 特征数据 除了标签以外的列读到x里面
y = data['Credit Default']  # 标签数据

st.subheader("标签数据分布")
st.write(y.value_counts())

# 填补缺失值，使用众数
for column in X.columns:
    if X[column].dtype in ['float64', 'int64']:
        mode = X[column].mode()[0]  # 计算计算该列的众数
        X[column].fillna(mode, inplace=True)  # 使用众数填补缺失值

# 分离连续特征和离散特征
continuous_features = X.select_dtypes(include=['float64', 'int64']).columns  # 连续
discrete_features = X.select_dtypes(include=['object']).columns  # 离散

# 连续特征标准化
scaler = StandardScaler()
X_continuous = scaler.fit_transform(X[continuous_features])  # 将连续特征进行标准化，均值为0，方差为1

# 离散特征编码
# 对于有序离散特征使用数值编码
label_encoder = LabelEncoder()
# 假设 'Years in current job' 是有序离散特征，将其编码为数值
X['Years in current job'] = label_encoder.fit_transform(X['Years in current job'])

# 对于无序离散特征使用独热编码
onehot_encoder = OneHotEncoder()
X_discrete = onehot_encoder.fit_transform(X[discrete_features.drop('Years in current job')])

# 组合处理后的连续特征和离散特征
X = pd.concat([pd.DataFrame(X_continuous, columns=continuous_features),
                pd.DataFrame(X_discrete.toarray(), columns=onehot_encoder.get_feature_names_out(discrete_features.drop('Years in current job')))], axis=1)

# 使用 train_test_split 函数按照 8:2 的比例划分数据集
# test_size=0.2 表示 20%的数据用作测试集，即验证集。
# random_state 是一个随机数种子，确保每次划分的结果相同，便于复现结果。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("处理后的训练集信息")
st.write(X_train.info())

st.subheader("训练集标签数据分布（处理前）")
st.write(y_train.value_counts())

# 对训练集进行过采样
from imblearn.over_sampling import SMOTE

# 使用 SMOTE 进行过采样,并赋值给自己
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

st.subheader("训练集标签数据分布（处理后）")
st.write(y_train.value_counts())

st.subheader("处理后的训练集信息（过采样后）")
st.write(X_train.info())

# 导入所需的库
import numpy as np  
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC  
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier  
import xgboost as xgb  
import lightgbm as lgb  

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report  
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt  # 绘图

if st.button("运行线性回归（错误示例）"):
    print("Linear Regression:")
    lr = LinearRegression()