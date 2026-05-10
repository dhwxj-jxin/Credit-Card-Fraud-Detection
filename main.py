import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


# 1. loading data

print("1. Loading Training and Test Data...")
train_df = pd.read_csv('fraudTrain1.csv')
test_df = pd.read_csv('fraudTest.csv')


# 2. Basic EDA 

print("2. Perfomring Basic EDA ...")

print("\n Raw Train Shape:", train_df.shape)
print(" Raw Test Shape:", test_df.shape)

print("\n Raw Data Info:")
print(train_df.info())

print("\n Missing Values (Raw):")
print(train_df.isnull().sum())

# Raw fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='is_fraud')
plt.title("Fraud Distribution (Raw Data)")
plt.show()

raw_fraud_rate = train_df['is_fraud'].mean() * 100
print(f"\n Raw Fraud Rate: {raw_fraud_rate:.4f}%")

# Raw summary statistics
print("\n Summary Statistics (Raw):")
print(train_df.describe())


# 3. Preprocessing

print("3. Preprocessing Data ...")

drop_cols = [
    'Unnamed: 0','cc_num','first','last','street','city','state',
    'job','dob','trans_num','merchant'
]
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Datetime features
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'], format='mixed')
test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'], format='mixed')

for df in [train_df, test_df]:
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month

train_df.drop(columns=['trans_date_trans_time'], inplace=True)
test_df.drop(columns=['trans_date_trans_time'], inplace=True)

# Encoding
label_cols = ['category', 'gender']
le = LabelEncoder()
for col in label_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


# 4. Advanced EDA 

print("4. Performing Advanced EDA...")

print("\n Cleaned Train Shape:", train_df.shape)
print(" Cleaned Test Shape:", test_df.shape)

# Fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='is_fraud')
plt.title("Fraud vs Non-Fraud Distribution (Post-Cleaning)")
plt.show()

fraud_rate = train_df['is_fraud'].mean() * 100
print(f"\n Fraud Rate After Cleaning: {fraud_rate:.4f}%")

# Correlation heatmap
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('is_fraud')
corr_target = train_df[numeric_cols].corrwith(train_df['is_fraud'], method='spearman')
top10 = corr_target.abs().sort_values(ascending=False).head(10)
top10_features = top10.index.tolist()

plt.figure(figsize=(8,6))
sns.heatmap(df[top10_features + ['is_fraud']].corr(method='spearman'),
            cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Top 10 Feature Correlations with Target (Spearman Heatmap)")
plt.tight_layout()
plt.show()

print("\nTOP 10 CORRELATIONS WITH is_fraud:")
print(corr_target.loc[top10_features].sort_values(ascending=False))

# Numeric feature distributions
num_cols = ['amt','city_pop','hour','day','month']
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# Category vs Fraud
plt.figure(figsize=(10,5))
sns.countplot(data=train_df, x='category', hue='is_fraud')
plt.title("Fraud by Category")
plt.xticks(rotation=45)
plt.show()

# Fraud vs hour
plt.figure(figsize=(8,4))
train_df.groupby('hour')['is_fraud'].mean().plot(marker='o')
plt.title("Fraud Rate by Hour")
plt.ylabel("Fraud Rate")
plt.grid(True)
plt.show()

figs = []
# Gender 
if 'gender' in train_df.columns:
    gender = train_df.groupby('gender')['is_fraud'].agg(['sum','count'])
    gender['fraud_rate'] = gender['sum'] / gender['count']
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.barplot(x=gender.index, y=gender['count'], ax=ax[0], palette='pastel')
    ax[0].set_title("Transaction Count by Gender")
    ax[0].set_xlabel("Gender (encoded)")
    sns.barplot(x=gender.index, y=gender['fraud_rate'], ax=ax[1], palette='rocket')
    ax[1].set_title("Fraud Rate by Gender")
    ax[1].set_xlabel("Gender (encoded)")
    plt.show()
else:
    print("No 'gender' column; skipping gender analysis.")


# Correlation comparison — NO-SMOTE vs SMOTE

#  No-SMOTE correlations
corr_no_smote = corr_target

# SMOTE 
X = train_df[numeric_cols]
y = train_df['is_fraud']

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)

corr_smote = pd.Series(
    [spearmanr(X_sm[col], y_sm)[0] for col in numeric_cols],
    index=numeric_cols
)

corr_compare = pd.DataFrame({
    "no_smote": corr_no_smote,
    "smote": corr_smote
}).sort_values("no_smote", ascending=False)

print("\nCORRELATION COMPARISON (NO-SMOTE vs SMOTE):")
print(corr_compare.head(15))

plt.figure(figsize=(10,6))
plt.plot(corr_compare["no_smote"].values, label="No-SMOTE", marker='o')
plt.plot(corr_compare["smote"].values, label="SMOTE", marker='s')
plt.xticks(range(len(corr_compare)), corr_compare.index, rotation=90)
plt.ylabel("Spearman Correlation with is_fraud")
plt.title("Correlation Shift After SMOTE")
plt.legend()
plt.tight_layout()
plt.show()


# 5. TRAIN-TEST Split

X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

print("5. Splitting X & y ...")
print(f"Training size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# Metrics
def evaluate_model(model_name, y_true, y_pred, y_probs):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_probs),
        'PR-AUC': auc(recall_vals, precision_vals)
    }


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title} Confusion Matrix")
    plt.show()


def plot_roc_pr(y_true, y_probs, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} ROC Curve")
    plt.grid(alpha=0.3)
    plt.show()

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(5,4))
    plt.plot(recall_vals, precision_vals, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} PR Curve")
    plt.grid(alpha=0.3)
    plt.show()


results = []

# 6. Tuning

tune_sample_size = 120000
X_tune = X_train.sample(tune_sample_size, random_state=42)
y_tune = y_train.loc[X_tune.index]

print(f"Tuning subset: {len(X_tune)} rows")

# ML MODELS 

# Logistic regression
print("6. Performing Logistic Regression...")
lr = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_params = {"C": np.logspace(-2, 2, 10), "class_weight": [None, "balanced"]}

lr_search = RandomizedSearchCV(lr, lr_params, n_iter=10, scoring='average_precision', cv=3, n_jobs=-1, random_state=42)
lr_search.fit(X_tune, y_tune)
best_lr = lr_search.best_estimator_
print("Best LR:", lr_search.best_params_)

# Predictions
y_prob_lr = best_lr.predict_proba(X_test)[:,1]
y_pred_lr = (y_prob_lr>=0.5).astype(int)
metrics_lr = evaluate_model("Logistic Reg (Tuned)", y_test, y_pred_lr, y_prob_lr)
results.append(metrics_lr)

plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
plot_roc_pr(y_test, y_prob_lr, "Logistic Regression")


# Decision tree 

# FEATURE SELECTION 
print("7. Performing Decision Tree Classifer...")
print("\n Feature Selection Justification for Decision Tree")

# Correlation with target (Spearman)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

corr_target = (
    train_df[num_cols + ['is_fraud']]
    .corr(method='spearman')['is_fraud']
    .drop('is_fraud')
    .abs()
    .sort_values(ascending=False)
)

print("\n Top features by absolute Spearman correlation with is_fraud:")
print(corr_target.head(10))

# Plot correlation 
plt.figure(figsize=(8,4))
corr_target.head(10).plot(kind='bar', color='#FF007F')
plt.title("Top 10 Features by Spearman Correlation with Fraud")
plt.ylabel("|Correlation|")
plt.grid(axis='y', alpha=0.3)
plt.show()


# fraud rate analysis 
print("\n Fraud Rate Checks")

if 'category' in train_df.columns:
    cat_fraud = (
        train_df.groupby('category')['is_fraud']
        .mean()
        .sort_values(ascending=False)
    )
    print("\n Fraud rate by category:")
    print(cat_fraud.head(10))

    plt.figure(figsize=(8,4))
    cat_fraud.head(10).plot(kind='bar', color='darkorange')
    plt.title("Fraud Rate by Category (Top 10)")
    plt.ylabel("Fraud Rate")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# shallow tree
print("\n Shallow Tree Feature Importance")

dt_probe = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)
dt_probe.fit(X_train, y_train)

probe_importance = (
    pd.Series(dt_probe.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
)

print("\n Top features by entropy-based importance:")
print(probe_importance.head(10))

plt.figure(figsize=(8,4))
probe_importance.head(10).plot(kind='barh', color='mediumseagreen')
plt.title("Top Features by Entropy-Based Importance (Shallow Tree)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.show()


# Final feature choices 
selected_features = ['amt', 'hour', 'category']

print("Final Selected Features for Decision Tree:")
for f in selected_features:
    print(f" - {f}")


available_features = [f for f in selected_features if f in X_train.columns]
print(f"Using features for visualization: {available_features}")

X_vis = X_train[available_features]
y_vis = y_train
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=3, random_state=42)
dt_entropy.fit(X_vis, y_vis)

plt.figure(figsize=(16, 10))
plot_tree(
    dt_entropy,
    feature_names=available_features,
    class_names=['Not Fraud', 'Fraud'],
    filled=True,
    rounded=True,
    fontsize=5,
)
plt.title("Decision Tree Visualization (Entropy & Info Gain)")
plt.show()

X_test_vis = X_test[available_features]
y_pred_dt = dt_entropy.predict(X_test_vis)
y_prob_dt = dt_entropy.predict_proba(X_test_vis)[:, 1]

evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt)
acc  = accuracy_score(y_test, y_pred_dt)
prec = precision_score(y_test, y_pred_dt, zero_division=0)
rec  = recall_score(y_test, y_pred_dt, zero_division=0)
f1   = f1_score(y_test, y_pred_dt, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob_dt)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_dt)
pr_auc = auc(recall_curve, precision_curve)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

#entropy confusion matrix 
plt.figure(figsize=(5,4))
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.title("Decision Tree (Entropy) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_dt)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--',color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#Precision-Recall Curve

plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve, label=f'PR AUC = {pr_auc:.3f}', color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#gini model
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_leaf=3, random_state=42)
dt_gini.fit(X_vis, y_vis)

plt.figure(figsize=(16, 10))
plot_tree(
    dt_gini,
    feature_names=available_features,
    class_names=['Not Fraud', 'Fraud'],
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title("Decision Tree Visualization (Gini)")
plt.show()

X_test_vis = X_test[available_features]
y_pred_dtg = dt_gini.predict(X_test_vis)
y_prob_dtg = dt_gini.predict_proba(X_test_vis)[:, 1]

evaluate_model("Decision Tree", y_test, y_pred_dtg, y_prob_dtg)
accg  = accuracy_score(y_test, y_pred_dt)
precg = precision_score(y_test, y_pred_dt, zero_division=0)
recg  = recall_score(y_test, y_pred_dt, zero_division=0)
f1g   = f1_score(y_test, y_pred_dt, zero_division=0)
roc_aucg = roc_auc_score(y_test, y_prob_dt)
precision_curveg, recall_curveg, _ = precision_recall_curve(y_test, y_prob_dt)
pr_auc = auc(recall_curveg, precision_curveg)
tng, fpg, fng, tpg = confusion_matrix(y_test, y_pred_dt).ravel()
print(f"TP: {tpg}, TN: {tng}, FP: {fpg}, FN: {fng}")

# Confusion Matrix

plt.figure(figsize=(5,4))
sns.heatmap([[tng, fpg], [fng, tpg]], annot=True, fmt='d', cmap='viridis', cbar=False,
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.title("Decision Tree (Gini) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

results.append(evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt))
print(results)

# validation curve

print("Generating Validation Curve...")

param_range = range(1, 21)
train_scores = []
test_scores = []
for depth in param_range:
    clf = DecisionTreeClassifier(
        max_depth=depth, 
        criterion='entropy', 
        random_state=42
    )
    
    clf.fit(X_vis, y_vis)
    train_pred = clf.predict(X_vis)
    train_f1 = f1_score(y_vis, train_pred)
    train_scores.append(train_f1)

    test_pred = clf.predict(X_test_vis)
    test_recall = recall_score(y_test, test_pred)
    test_scores.append(test_recall)


plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores, label="Training Recall", color="blue", marker="o")
plt.plot(param_range, test_scores, label="Test Recall", color="red", marker="o")
plt.title("Validation Curve: Effect of Tree Depth on Performance")
plt.xlabel("Max Depth of Tree")
plt.ylabel("Recall")
plt.legend(loc="best")
plt.xticks(param_range)
plt.grid(True, alpha=0.3)

best_depth_idx = np.argmax(test_scores)
best_depth = param_range[best_depth_idx]
best_score = test_scores[best_depth_idx]

plt.axvline(x=best_depth, color='green', linestyle='--', label=f'Optimal Depth ({best_depth})')
plt.text(best_depth + 0.5, best_score, f'Peak Recall: {best_score:.4f}', color='green')
plt.axvspan(param_range[0], best_depth, color='purple', alpha=0.15)
plt.axvspan(best_depth, param_range[-1], color='gold', alpha=0.15)

plt.show()

print(f"Validation Curve Generated. Optimal Depth found: {best_depth}")

# xgboost
print(f"8. XGBoost ML Model ... ")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,        
    learning_rate=0.1,       
    max_depth=6,             
    subsample=0.8,           
    colsample_bytree=0.8,    
    random_state=42,
    eval_metric='logloss',   
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred_xgb)
prec = precision_score(y_test, y_pred_xgb)
rec  = recall_score(y_test, y_pred_xgb)
f1   = f1_score(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, y_prob_xgb)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_auc = auc(recall_curve, precision_curve)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()

# Confusion Matrix

plt.figure(figsize=(5,4))
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.title('XGBoost Confusion Matrix')
plt.show()

# ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# PR Curve

plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve, label=f'PR AUC = {pr_auc:.4f}', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('XGBoost Precision–Recall Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Feature Importance 

xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain', title="Top 10 Important Features")
plt.show()

results.append(evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb))

#  RANDOM FOREST 

print("\n 9. Performing Random Forest Classifier ...")
rf = RandomForestClassifier(class_weight='balanced')
rf_params = {
    "n_estimators":[200,300,500],
    "max_depth":[8,12,16,None],
    "min_samples_split":[2,10,50],
    "min_samples_leaf":[1,5,10]
}

rf_search = RandomizedSearchCV(rf, rf_params, n_iter=15, scoring='average_precision', cv=3, n_jobs=-1, random_state=42)
rf_search.fit(X_tune, y_tune)
best_rf = rf_search.best_estimator_
print("Best RF:", rf_search.best_params_)

y_prob_rf = best_rf.predict_proba(X_test)[:,1]
y_pred_rf = (y_prob_rf>=0.5).astype(int)
results.append(evaluate_model("Random Forest (Tuned)", y_test, y_pred_rf, y_prob_rf))

plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plot_roc_pr(y_test, y_prob_rf, "Random Forest")

# SVM 

from sklearn.svm import SVC                             
from sklearn.pipeline import Pipeline    
print("\n 10. SVM (RBF Kernel) Model ...")

svm_sample_size = 50000 
if len(X_train) > svm_sample_size:
    svm_X_train = X_train.sample(svm_sample_size, random_state=42)
    svm_y_train = y_train.loc[svm_X_train.index]
else:
    svm_X_train = X_train
    svm_y_train = y_train

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42))
])

svm_pipeline.fit(svm_X_train, svm_y_train)
y_prob_svm = svm_pipeline.predict_proba(X_test)[:, 1]
y_pred_svm = (y_prob_svm >= 0.5).astype(int)

results.append(evaluate_model("SVM (RBF)", y_test, y_pred_svm, y_prob_svm))

print("\n SVM Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_svm, "SVM (RBF)")

print("\n SVM ROC & PR Curves")
plot_roc_pr(y_test, y_prob_svm, "SVM (RBF)")

# LIGHTGBM 
print("\n 11. Performing LightGBM...")
lgb_base = lgb.LGBMClassifier(class_weight='balanced')

lgb_params = {
    "n_estimators":[200,400,600],
    "learning_rate":[0.01,0.05,0.1],
    "num_leaves":[31,63,127],
    "subsample":[0.6,0.8,1.0]
}

lgb_search = RandomizedSearchCV(lgb_base, lgb_params, n_iter=15, scoring='average_precision', cv=3, n_jobs=-1, random_state=42)
lgb_search.fit(X_tune, y_tune)
best_lgb = lgb_search.best_estimator_
print("Best LGBM:", lgb_search.best_params_)

y_prob_lgb = best_lgb.predict_proba(X_test)[:,1]
y_pred_lgb = (y_prob_lgb>=0.5).astype(int)
results.append(evaluate_model("LightGBM (Tuned)", y_test, y_pred_lgb, y_prob_lgb))

plot_confusion_matrix(y_test, y_pred_lgb, "LightGBM")
plot_roc_pr(y_test, y_prob_lgb, "LightGBM")

# SHAP for xgboost

print("\n 12. Performing SHAP for XGBoost...")

shap_sample = X_train.sample(3000, random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(shap_sample)

shap.summary_plot(shap_values, shap_sample, plot_type="bar")

example_idx = y_test[y_test==1].index[0]
example = X_test.loc[[example_idx]]
shap.force_plot(explainer.expected_value, explainer.shap_values(example), example, matplotlib=True)

# LIME for xgboost

print("\n 13. Performing LIME ...")

lime_exp = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['Not Fraud','Fraud'],
    mode='classification'
)

i = 10
exp = lime_exp.explain_instance(X_test.iloc[i], xgb_model.predict_proba, num_features=10)

print("\nTop LIME features:")
for feat, weight in exp.as_list():
    print(f"{feat}: {weight}")


# 13. Final Metrics Summary Table

print("\n============= FINAL MODEL PERFORMANCE SUMMARY =============")
results_df = pd.DataFrame(results)
print(results_df)

ax = results_df.set_index('Model').plot(kind='bar', figsize=(12,6))
plt.title("Model Comparison (Tuned)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(alpha=0.3)

plt.xticks(rotation=45, ha='right')   
plt.tight_layout()
plt.show()

