#The first thing to do is load the Wisconsin Breast Cancer sample dataset into the pandas data frame
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
#The second step is to build the binary decision tree
X = data.data
y = data.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier,export_text
model = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=2, criterion='gini', random_state=42)
model.fit(X_train, y_train)
# Compute the classification report, including precision, recall, and F1 score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred)
print("Classification report:")
print(report)
#Data after dimensionality reduction using PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = DecisionTreeClassifier(min_samples_leaf=2,min_samples_split=5,max_depth=2,criterion='gini',random_state=42)
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print("\nSecond model (PCA dimensionality reduction)：")
print(f"Model accuracy: {accuracy_pca:.4f}")
report_pca = classification_report(y_test_pca, y_pred_pca)
print("Classification report:")
print(report_pca)
#Principal Component Analysis (PCA) is used for dimensionality reduction, keeping the first two principal components
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = DecisionTreeClassifier(min_samples_leaf=2,min_samples_split=5,max_depth=2,criterion='gini',random_state=42)
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print("\nThe third model (PCA to reduce the dimensionality of the two principal components) after the data")
print(f"Model accuracy: {accuracy_pca:.4f}")
report_pca = classification_report(y_test_pca, y_pred_pca)
print("Classification report:")
print(report_pca)
cm = confusion_matrix(y_test_pca, y_pred_pca)
print("confusion matrix:")
print(cm)
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]
FPR = FP / (FP + TN)
TPR = TP / (TP + FN)
print(f"FP: {FP}")
print(f"TP: {TP}")
print(f"FPR: {FPR:.4f}")
print(f"TPR: {TPR:.4f}")