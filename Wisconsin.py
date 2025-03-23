# The first thing to do is load the Wisconsin Breast Cancer sample dataset into the pandas data frame
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_names = [
    'id', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'
]
df = pd.read_csv(url, names=column_names)
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
df['class'] = df['class'].apply(lambda x: 0 if x == 2 else 1)

# The second step is to build the binary decision tree
X = df.drop(['id', 'class'], axis=1).values
y = df['class'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=2, criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("report:")
print(classification_report(y_test, y_pred))
import numpy as np

# Function to calculate entropy
def Calculating_entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])
    return entropy

initial_entropy = Calculating_entropy(y_train)
print(f"Initial Entropy: {initial_entropy}")

tree = model.tree_
first_split_feature = tree.feature[0]
first_split_threshold = tree.threshold[0]
print(f"First split feature: {column_names[first_split_feature + 1]}")  # +1是因为X中不包含id和class列
print(f"First split threshold: {first_split_threshold}")

X_train_split_left = X_train[X_train[:, first_split_feature] <= first_split_threshold]
y_train_split_left = y_train[X_train[:, first_split_feature] <= first_split_threshold]
X_train_split_right = X_train[X_train[:, first_split_feature] > first_split_threshold]
y_train_split_right = y_train[X_train[:, first_split_feature] > first_split_threshold]

entropy_left = Calculating_entropy(y_train_split_left)
entropy_right = Calculating_entropy(y_train_split_right)
weighted_entropy = (len(y_train_split_left) / len(y_train)) * entropy_left + (len(y_train_split_right) / len(y_train)) * entropy_right
print(f"Entropy after first split: {weighted_entropy}")

# Function to calculate gini index
def Calculating_gini_index(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    gini_index = 1 - np.sum(ps ** 2)
    return gini_index

initial_gini_index = Calculating_gini_index(y_train)
print(f"Initial Gini Index: {initial_gini_index}")

gini_index_left = Calculating_gini_index(y_train_split_left)
gini_index_right = Calculating_gini_index(y_train_split_right)
weighted_gini_index = (len(y_train_split_left) / len(y_train)) * gini_index_left + (len(y_train_split_right) / len(y_train)) * gini_index_right
print(f"Gini Index after first split: {weighted_gini_index}")

# Function to calculate misclassification error
def Calculating_misclassification(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    misclassification = 1 - np.max(ps)
    return misclassification

initial_misclassification = Calculating_misclassification(y_train)
print(f"Initial Misclassification Error: {initial_misclassification}")

misclassification_left = Calculating_misclassification(y_train_split_left)
misclassification_right = Calculating_misclassification(y_train_split_right)
weighted_misclassification = (len(y_train_split_left) / len(y_train)) * misclassification_left + (len(y_train_split_right) / len(y_train)) * misclassification_right
print(f"Misclassification Error after first split: {weighted_misclassification}")

information_gain = initial_entropy - weighted_entropy
print(f"Information Gain after first split: {information_gain}")