import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve, precision_score,f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import joblib

dataFrame = pd.read_csv('e-shop-clothing.csv')

labelEncoder = LabelEncoder()
dataFrame['page 2 (clothing model)'] = labelEncoder.fit_transform(dataFrame['page 2 (clothing model)'])

numerical_columns = ['order', 'price']
Q1 = dataFrame[numerical_columns].quantile(0.25)
Q3 = dataFrame[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

condition = (dataFrame[numerical_columns] >= lower_bound) & (dataFrame[numerical_columns] <= upper_bound)
dataFrame = dataFrame.loc[condition['order']]
dataFrame = dataFrame.loc[condition['price']]

dataFrame = dataFrame.reset_index(drop=True)

numerical_df = dataFrame[['order','price']]
categorical_df = dataFrame[['month', 'day', 'country', 'page 1 (main category)', 'colour','location','model photography','page']]

X = pd.concat([numerical_df, categorical_df], axis=1)
y = dataFrame['price 2'] - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC(kernel='linear', random_state=5805)

param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': ['scale','auto']}
try:
    grid_search = joblib.load('svm_model.pkl')
    print("Loaded checkpoints successfully...")

except FileNotFoundError as e:
    print("No previous checkpoint found. Starting new training...")
    grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5, verbose=2)
    grid_search.fit(X_train, y_train)

print("\nThe best parameter for SVM (linear kernel) is: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print("\nThe test accuracy of SVM (linear kernel) is: ", accuracy_score(y_test, y_pred))
print("\nThe confusion matrix of SVM (linear kernel) is: \n", confusion_matrix(y_test, y_pred))
print("\nThe precision of SVM (linear kernel) is: ", precision_score(y_test,y_pred))
print("\nThe recall of SVM (linear kernel) is: ", recall_score(y_test,y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print("\nThe specificity of SVM (linear kernel) is: ", specificity)
print("\nThe f1 score of SVM (linear kernel) is: ", f1_score(y_test, y_pred))

scaler = StandardScaler()
X_CV = scaler.fit_transform(X)
X_CV = pd.DataFrame(X_CV, columns=X.columns)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
print("\nThe accuracy for each fold is: ")
for train_index, test_index in skf.split(X_CV, y):
    X_train_k, X_test_k = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
    best_model.fit(X_train_k, y_train_k)
    accuracy = best_model.score(X_test_k, y_test_k)
    print("Accuracy:", accuracy)

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, linestyle='--', label='SVM (linear kernel) AUC = %0.2f' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

joblib.dump(grid_search, 'svm_model.pkl')




