# PRODIGY_DS_03
## Task-03: Decision Tree Classifier

### Objective
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

### Dataset
- Source: Bank Marketing dataset (UCI Machine Learning Repository)
- File used: `bank.csv` (sample dataset with ~4,500 rows)
- Note: A larger version (`bank-full.csv`, ~45,000 rows) is also available for deeper analysis.
- Target variable: `y` (whether the client subscribed to a term deposit)

### Tools Used
- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)

### Steps
1. Load and explore the dataset (`sep=";"` required for proper parsing).
2. Preprocess data:
   - Encode categorical variables using one-hot encoding.
   - Split into training and testing sets.
3. Train a Decision Tree Classifier (`max_depth=5`).
4. Evaluate model performance using accuracy, confusion matrix, and classification report.
5. Visualize the decision tree and confusion matrix.

### Code
See `Task03.py` for full implementation.

### Visualizations
#### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

#### Decision Tree
![Decision Tree](outputs/decision_tree.png)

### Insights
- The decision tree highlights key demographic and behavioral features influencing purchase decisions.
- Features such as `age`, `job`, `marital status`, and `education` often appear near the top of the tree.
- The confusion matrix shows that the model predicts "no purchase" more accurately than "yes purchase," reflecting class imbalance in the dataset.
- Accuracy is moderate (~70–80%), but interpretability is high — you can see exactly how decisions are made.

### Conclusion
Decision trees provide a transparent and interpretable model for customer purchase prediction.  
While accuracy can be improved with ensemble methods (Random Forest, Gradient Boosting), this task demonstrates how demographic and behavioral data can be leveraged for marketing analytics and how decision trees reveal the most influential features.
