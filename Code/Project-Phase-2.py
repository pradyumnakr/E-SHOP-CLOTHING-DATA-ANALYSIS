import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 15)

dataFrame = pd.read_csv('e-shop-clothing.csv')

labelEncoder = LabelEncoder()
dataFrame['page 2 (clothing model)'] = labelEncoder.fit_transform(dataFrame['page 2 (clothing model)'])

num_outlier_columns = ['order', 'price']
Q1 = dataFrame[num_outlier_columns].quantile(0.25)
Q3 = dataFrame[num_outlier_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
condition = (dataFrame[num_outlier_columns] >= lower_bound) & (dataFrame[num_outlier_columns] <= upper_bound)
dataFrame = dataFrame.loc[condition['order']]
dataFrame = dataFrame.loc[condition['price']]

dataFrame = dataFrame.reset_index(drop=True)

numerical_df = dataFrame[['year', 'session ID', 'order']]
categorical_df = dataFrame[['month', 'day', 'country', 'page 1 (main category)','page 2 (clothing model)', 'colour','location','model photography','page','price 2']]

combined_df = pd.concat([numerical_df, categorical_df], axis=1)

X = combined_df.drop(columns=['year','session ID', 'page 2 (clothing model)'])
y = dataFrame[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True)

scaler = StandardScaler()

standard_train = scaler.fit_transform(X_train)
standard_test = scaler.transform(X_test)

standardized_train = pd.DataFrame(standard_train, columns=X.columns)
standardized_test = pd.DataFrame(standard_test, columns=X.columns)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

scaler = StandardScaler()
standardized_y_train = scaler.fit_transform(y_train)
standardized_y_test = scaler.transform(y_test)

def backward_stepwise_regression(X, y, threshold=0.01):
    num_features = X.shape[1]
    included_features = list(range(num_features))
    results = []

    while True:
        model = sm.OLS(y, sm.add_constant(X.iloc[:, included_features])).fit()
        p_values = model.pvalues[1:]
        print(model.summary())
        max_p_value = p_values.max()
        if max_p_value > threshold:
            excluded_feature_idx = p_values.idxmax()
            idx = X.columns.to_list().index(excluded_feature_idx)
            included_features.remove(idx)
            results.append((excluded_feature_idx, max_p_value, model.aic, model.bic, model.rsquared_adj))
        else:
            break

    return included_features, results, model


selected_features, results, model = backward_stepwise_regression(standardized_train, standardized_y_train)

columns = ["Eliminated Feature", "P-value", "AIC", "BIC", "Adjusted R-squared"]
df_results = pd.DataFrame(results, columns=columns)

print("\nBackward Stepwise Regression Results:\n")
print(df_results)

print("\nFinal Selected Features:")
print(standardized_train.columns[selected_features])

X_test_subset = standardized_test.iloc[:, selected_features]
X_test_model = sm.add_constant(X_test_subset)
y_pred = model.predict(X_test_model)

y_temp = y_pred * y_train.std()[0]
y_pred_reverse = y_temp + y_train.mean()[0]

plt.scatter(range(1000), y_test[:1000], label='Original Test Set', color='blue')
plt.scatter(range(1000), y_pred_reverse[:1000], label='Predicted Values', color='red')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.title('Comparison of Original Test Set and Predicted Values')
plt.show()

plt.plot(y_test[:1000], label='Original Test Set')
plt.plot(y_pred_reverse[:1000], label='Predicted Values')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.title('Comparison of Original Test Set and Predicted Values')
plt.show()

mse = mean_squared_error(y_pred, standardized_y_test)
print("\nMean Squared Error (MSE):", mse)

feature_names = standardized_test.iloc[:, selected_features].columns
coefficients = model.params[1:]

final_equation = ''
for feature, coefficient in zip(feature_names, coefficients):
    final_equation += f" + ({coefficient} * {feature})"
print("\nThe final equation is: ", final_equation[3:])

comparison = pd.DataFrame({'Price': y_test.values.flatten(), 'Predicted_Price': y_pred_reverse})
print("\nComparison of Original Test Set and Predicted Values\n")
print(comparison.head())

final_results = []
final_results.append((model.rsquared, model.rsquared_adj, model.aic, model.bic, mse))
final_columns = ['Rsquared', 'Rsquared-Adj', 'AIC', 'BIC', 'MSE']
final_model = pd.DataFrame(final_results, columns=final_columns)
print("\nThe final model is: \n", str(final_model))

confidence_intervals = model.conf_int()
print("\nThe confidence intervals are: \n", confidence_intervals)



