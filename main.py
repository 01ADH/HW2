import pandas as pd
from sklearn.datasets import load_iris
#The first step is to load the iris dataset into a pandas data frame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
X = df.drop('target', axis=1)  # 特征
y = df['target']
from sklearn.metrics import recall_score, precision_score, f1_score
#Define a list of evaluation metrics
recall_scores = []
precision_scores = []
f1_scores = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#In the third step we iterate the depth from 1 to 5
for depth in range(1, 6):
    X_train, X_test, y_train, y_test = train_test_split(X, y,)
    model = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)
    print(f"Depth: {depth}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
results = pd.DataFrame({
    'Depth': range(1, 6),
    'Recall': recall_scores,
    'Precision': precision_scores,
    'F1 Score': f1_scores
})
print("\nResults:\n", results)
import matplotlib.pyplot as plt
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), recall_scores, marker='o', label='Recall')
plt.plot(range(1, 6), precision_scores, marker='o', label='Precision')
plt.plot(range(1, 6), f1_scores, marker='o', label='F1 Score')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Score')
plt.title('Evaluation Metrics vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()




