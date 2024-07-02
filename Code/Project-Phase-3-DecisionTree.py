import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve, precision_score,f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

clf = DecisionTreeClassifier(random_state=5805)
tuned_parameters = [{'max_depth': range(2,15,1),
                     'min_samples_split': range(2,10,1),
                     'min_samples_leaf': range(1,5,1),
                     'max_features': range(1,5,1),
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy', 'log_loss']}]

try:
    gridSearch = joblib.load('decision_tree_pre.pkl')
    print("\nLoaded checkpoints successfully")

except FileNotFoundError:
    print("No previous checkpoint found. Starting new training...")
    gridSearch = GridSearchCV(clf, tuned_parameters, cv=5, scoring='accuracy',verbose=2)
    gridSearch.fit(X_train,y_train)

print('\nBest parameters set found on development', gridSearch.best_params_)
best_estimator = gridSearch.best_estimator_
y_pred = best_estimator.predict(X_test)

print("\nThe accuracy of the pre-pruned model is: ", accuracy_score(y_test, y_pred))
print("\nThe confusion matrix is: \n", confusion_matrix(y_test, y_pred))
print("\nThe precision of the pre-pruned model is: ", precision_score(y_test, y_pred))
print("\nThe recall of the pre-pruned model", recall_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print("\nThe specificity of the pre-pruned model is: ", specificity)
print("\nThe f1 score of the pre-pruned model is: ", f1_score(y_test, y_pred))

scaler = StandardScaler()
X_CV = scaler.fit_transform(X)
X_CV = pd.DataFrame(X_CV, columns=X.columns)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
print("\nThe accuracy for each fold is: ")
for train_index, test_index in skf.split(X_CV, y):
    X_train_k, X_test_k = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
    best_estimator.fit(X_train_k, y_train_k)
    accuracy = best_estimator.score(X_test_k, y_test_k)
    print("Accuracy:", accuracy)

fpr, tpr, _ = roc_curve(y_test,y_pred)
plt.plot(fpr, tpr, linestyle='--', label='Pre-Pruned Decision Tree AUC = %0.2f ' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True positive rate')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=best_estimator.classes_)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

plt.figure(figsize=(16,8))
plot_tree(best_estimator, rounded=True, filled=True)
plt.show()

joblib.dump(gridSearch, 'decision_tree_pre.pkl')

##Post Pruned Decision Tree

clf = DecisionTreeClassifier(random_state=5805)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(16,8))
plt.plot(ccp_alphas, test_scores)
plt.grid(True)
plt.xlabel('Effective Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy VS Effective Alpha')
plt.show()

optimalAlpha = ccp_alphas[np.argmax(test_scores)]
print("\nOptimal Alpha: ", optimalAlpha)

ppclf = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimalAlpha)
ppclf.fit(X_train, y_train)
plt.figure(figsize=(16,8))
plot_tree(ppclf, rounded=True, filled=True)
plt.show()

y_pp_pred = ppclf.predict(X_test)
print('\n The accuracy of the post-pruned model is: ', accuracy_score(y_test, y_pp_pred))
print('\n The confusion matrix is: \n', confusion_matrix(y_test, y_pp_pred))
print('\n The precision score of the post-pruned model is: ', precision_score(y_test, y_pp_pred))
print('\n The recall score of the post-pruned model is: ', recall_score(y_test, y_pp_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pp_pred).ravel()
specificity = tn / (tn + fp)
print('\nThe specificity of the post-pruned model is: ', specificity)
print('\nThe f1 score of the post-pruned model is: ', f1_score(y_test, y_pp_pred))

print("\nThe accuracy for each fold is: ")
for train_index, test_index in skf.split(X_CV, y):
    X_train_k, X_test_k = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
    ppclf.fit(X_train_k, y_train_k)
    accuracy = ppclf.score(X_test_k, y_test_k)
    print("Accuracy:", accuracy)

fpr, tpr, _ = roc_curve(y_test, y_pp_pred)
plt.plot(fpr, tpr, linestyle='--', label='Post-Pruned Decision Tree AUC = %0.2f ' % roc_auc_score(y_test, y_pp_pred))
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pp_pred, labels=ppclf.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=ppclf.classes_)
disp.plot()
plt.show()


