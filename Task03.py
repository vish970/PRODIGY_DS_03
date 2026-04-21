# Task03.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("C:/Users/Vishal.S/Downloads/intership of google play store/prodigy internship/prodigy_DS_01/bank.csv", sep=";")

print("Dataset shape:", data.shape)
print(data.head())
data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop("y_yes", axis=1)  # target column after encoding
y = data_encoded["y_yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No","Yes"], yticklabels=["No","Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["No","Yes"], filled=True)
plt.savefig("outputs/decision_tree.png")
plt.close()

print("Outputs saved in outputs/ folder.")
