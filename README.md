# machine-lerning_project_simplilearn
##Machine learning course end project at simplilearn
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("HR_comma_sep_1.csv")
df.head()
df.info()
df.isnull().sum().sum()
df.describe(exclude=np.number)
hp = df.drop(columns = ['Department', 'salary'])
sns.heatmap(hp.corr(), xticklabels=hp.columns, yticklabels=hp.columns)
sns.displot(df.satisfaction_level, kde=True)
sns.displot(df.last_evaluation, kde=True)
sns.displot(df.average_montly_hours, kde=True)
sns.barplot(x= df.left,y=df.number_project, hue=df.number_project)
df_k = df.filter(['satisfaction_level', 'last_evaluation', 'left'],axis=1)
df_k.info()
df_k['satisfaction_level'].unique()
len(df_k['satisfaction_level'].unique())
df_k['last_evaluation'].unique()
len(df_k['last_evaluation'].unique())
df_k['left'].unique()
array([1, 0], dtype=int64)
len(df_k['left'].unique())
X = df_k
y = df_k['left']
cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=1) 
kmeans.fit(X)
kmeans.inertia_
kmeans.cluster_centers_
df_k['cluster'] = kmeans.labels_
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
df1= df.drop(['salary', 'Department'], axis='columns')
df1.info()
c = df.filter(['salary', 'Department'],axis=1)
c
c = pd.get_dummies(c, columns=['salary', 'Department'])
c.info()
df1= df1.join(c, lsuffix='_caller', rsuffix='_other')
df1
df1.info()
from sklearn.model_selection import train_test_split
y = df1['left']
X = df1.drop(['left'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
Before OverSampling, counts of label '1': 2862
Before OverSampling, counts of label '0': 9137 
from imblearn.over_sampling import SMOTE
X_train.info()
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
df_kf = df.copy()
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
df_kf['salary'] = Le.fit_transform(df_kf['salary'])
df_kf['Department'] = Le.fit_transform(df_kf['Department'])
df_kf.info()
X = df_kf.drop('left', axis = 1)
y = df_kf['left']
kf =KFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1
from sklearn import linear_model
for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
    cvs = cross_val_score(linear_model.LogisticRegression(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error").mean()
    print(score + " : "+ str(cvs)) 
from sklearn import ensemble
for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
    cvs = cross_val_score(ensemble.RandomForestRegressor(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error").mean()
    print(score + " : "+ str(cvs))
for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
    cvs = cross_val_score(ensemble.GradientBoostingClassifier(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error").mean()
    print(score + " : "+ str(cvs)) 
df_roc = df.copy()
Le = LabelEncoder()
df_roc['salary'] = Le.fit_transform(df_roc['salary'])
df_roc['Department'] = Le.fit_transform(df_roc['Department'])
X = df_roc.drop('left', axis = 1)
y = df_roc['left']
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state=42)
logr = linear_model.LogisticRegression()
logr.fit(X_train,y_train)
LogisticRegression()
y_pred_logr  = logr.predict(X_test)
from sklearn.metrics import roc_auc_score
roc_auc_logr = roc_auc_score(y_test, y_pred_logr)
roc_auc_logr
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(logr, X_test, y_test)
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_logr))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_logr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
RandomForestClassifier()
y_pred_rfc = rfc.predict(X_test)
roc_auc_rfc = roc_auc_score(y_test, y_pred_rfc)
roc_auc_rfc
RocCurveDisplay.from_estimator(rfc, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred_rfc))
cm2 = confusion_matrix(y_test, y_pred_rfc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot()
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred_gbc = gbc.predict(X_test)
roc_auc_gbc = roc_auc_score(y_test, y_pred_gbc)
roc_auc_gbc
RocCurveDisplay.from_estimator(gbc, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred_gbc))
cm3 = confusion_matrix(y_test, y_pred_gbc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp.plot()
plt.show()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred_logr)
fpr, tpr, thresholds = roc_curve(y_test, logr.predict_proba(X_test)[:,1])
rf_roc_auc = roc_auc_score(y_test, y_pred_rfc)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
gbc_roc_auc = roc_auc_score(y_test, y_pred_gbc)
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingClassifier (area = %0.2f)' % gbc_roc_auc)
plt.plot([0, 1], [0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()
feature_labels = np.array(['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'])
importance = rfc.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))
department_hr-0.10%
department_RandD-0.69%
salary_high-0.78%
department_management-1.22%
last_evaluation-12.13%
Work_accident-14.11%
promotion_last_5years-18.33%
time_spend_company-18.51%
satisfaction_level-34.13%
