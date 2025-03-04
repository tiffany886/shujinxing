import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import shap
import matplotlib.pyplot as plt

# 定义函数来处理数据
def preprocess_data(data):
    # 去除Id列
    data = data.drop(columns=['Id'])
    
    # 分离特征数据和标签数据
    X = data.drop(['Credit Default'], axis=1)
    y = data['Credit Default']
    
    # 填补缺失值，使用众数
    for column in X.columns:
        if X[column].dtype in ['float64', 'int64']:
            mode = X[column].mode()[0]
            X[column].fillna(mode, inplace=True)
    
    # 分离连续特征和离散特征
    continuous_features = X.select_dtypes(include=['float64', 'int64']).columns
    discrete_features = X.select_dtypes(include=['object']).columns
    
    # 连续特征标准化
    scaler = StandardScaler()
    X_continuous = scaler.fit_transform(X[continuous_features])
    
    # 离散特征编码
    label_encoder = LabelEncoder()
    X['Years in current job'] = label_encoder.fit_transform(X['Years in current job'])
    
    onehot_encoder = OneHotEncoder()
    X_discrete = onehot_encoder.fit_transform(X[discrete_features.drop('Years in current job')])
    
    # 组合处理后的连续特征和离散特征
    X = pd.concat([pd.DataFrame(X_continuous, columns=continuous_features),
                   pd.DataFrame(X_discrete.toarray(), columns=onehot_encoder.get_feature_names_out(discrete_features.drop('Years in current job')))], axis=1)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 对训练集进行过采样
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

# 定义函数来训练和评估模型
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    return auc_score

# Streamlit应用程序
def main():
    st.title("信贷预测案例研究")
    
    # 上传数据文件
    # uploaded_file = st.file_uploader("上传数据文件 (CSV格式)", type="csv")
    uploaded_file=True
    if uploaded_file is not None:
        # 读取数据
        # data = pd.read_csv(uploaded_file)
        data = pd.read_csv("C:\\Users\\Tiffany\\Desktop\\python\\shujinxing\\项目一信贷风险预测\\data.csv")
        # 显示数据前几行
        st.subheader("数据预览")
        st.write(data.head())
        
        # 数据预处理
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        # 定义模型及其最佳参数
        models = {
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(C=10, gamma='auto', kernel='rbf', probability=True),
            "Naive Bayes": GaussianNB(var_smoothing=1e-5),
            "Random Forest": RandomForestClassifier(
                max_depth=10, max_features='sqrt', min_samples_leaf=1,
                min_samples_split=10, n_estimators=200
            ),
            "XGBoost": xgb.XGBClassifier(
                colsample_bytree=0.8, learning_rate=0.1, max_depth=3,
                n_estimators=100, subsample=0.8
            ),
            "LightGBM": lgb.LGBMClassifier(
                colsample_bytree=0.8, learning_rate=0.1, max_depth=5,
                n_estimators=100, num_leaves=31, subsample=0.8
            )
        }
        
        # 训练和评估模型
        st.subheader("模型评估")
        results = {}
        for name, model in models.items():
            auc_score = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
            results[name] = auc_score
            st.write(f"{name}: AUC = {auc_score:.3f}")
        
        # 绘制ROC曲线
        st.subheader("ROC曲线")
        plt.figure(figsize=(12, 6))
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_test = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_test)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test Set ROC Curves')
        plt.legend(loc="lower right")
        st.pyplot(plt)
        
        # 可解释性分析
        st.subheader("可解释性分析 (LightGBM)")
        best_params = {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 63}
        lgb_best = lgb.LGBMClassifier(**best_params)
        lgb_best.fit(X_train, y_train)
        
        explainer = shap.Explainer(lgb_best)
        shap_values = explainer.shap_values(X_train)
        
        # SHAP 汇总图设置为条形图，可以显示特征重要性
        shap.summary_plot(shap_values[1], X_train, plot_type="bar")
        st.pyplot(plt)

if __name__ == "__main__":
    main()