import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve, precision_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

parameters = [{'n_neighbors':range(1,11,1),
              'p':[1,2],
              'weights' : ['uniform','distance'],
               'algorithm' : ['auto','ball_tree','kd_tree']}]
clf = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, scoring='accuracy',verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters for the KNN classifier is: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print("\nThe accuracy of the KNN classifier model is: ", accuracy_score(y_test,y_pred))
print("\nThe confusion matrix of the KNN classifier is : \n", confusion_matrix(y_test,y_pred))
print("\nThe precision of the KNN classifier is : ", precision_score(y_test,y_pred))
print("\nThe recall of the KNN classifier is : ", recall_score(y_test,y_pred))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
specificity = tn / (tn + fp)
print("\nThe specificity of the KNN classifier is : ", specificity)
print("\nThe f1 score of the KNN classifier is : ", f1_score(y_test,y_pred))

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

fpr, tpr, _ = roc_curve(y_test,y_pred)
plt.plot(fpr, tpr,linestyle='--',label='KNN Classifier AUC = %0.2f' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test,y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


