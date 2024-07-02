import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 15)

dataFrame = pd.read_csv('e-shop-clothing.csv')
print("The shape of the dataframe is:", dataFrame.shape)
print("\n")
print("There are ", str(dataFrame.shape[0]), "observations")
print("\n")
print("There are ", str(dataFrame.shape[1]), "features")
print("\n")
print("Information about the features:\n", dataFrame.info())
print("\n")
print("More info about the dataframe:\n", str(dataFrame.describe()))
print("\n")
print("The missing values in the dataframe is:\n" + str(dataFrame.isnull().sum()))
print("\n")
print("\nNumber of duplicate rows:", dataFrame.duplicated().sum())

monthlysales = dataFrame.groupby('month').agg({'order':'sum'})
plt.figure(figsize = (10,6))
monthlysales.plot(color = 'skyblue', kind='bar')
plt.title("Monthly Orders")
plt.xlabel("Month")
plt.xticks(rotation = 45)
plt.grid(linestyle = 'dotted',alpha=0.7, axis='y')
plt.ylabel("Total Orders")
plt.tight_layout()
plt.show()

colorNames = {
    1: 'beige',
    2: 'black',
    3: 'blue',
    4: 'brown',
    5: 'burgundy',
    6: 'gray',
    7: 'green',
    8: 'navy blue',
    9: 'of many colors',
    10: 'olive',
    11: 'pink',
    12: 'red',
    13: 'violet',
    14: 'white'
}

avgPricePerColor = dataFrame.groupby('colour').agg({"price":"mean"})
plt.figure(figsize = (10,6))
avgPricePerColor.plot(color='orange', kind='bar', alpha=0.7)
plt.title("Average price per color")
plt.xlabel("Color")
plt.ylabel("Avg Price")
plt.xticks(range(len(avgPricePerColor)), [colorNames[int(colour)] for colour in avgPricePerColor.index])
plt.grid(linestyle = '--',alpha=0.7, axis='y')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

downSampledData = dataFrame.sample(n=60000,random_state=5805).reset_index(drop=True)

labelEncoder = LabelEncoder()
dataFrame['page 2 (clothing model)'] = labelEncoder.fit_transform(dataFrame['page 2 (clothing model)'])

price = dataFrame[['price', 'price 2']].copy()

plt.figure(figsize=(10,6))
sns.boxplot(data=dataFrame, x='order',)
plt.title("Box Plot of Order")
plt.xlabel('Values')

plt.figure(figsize = (10,6))
sns.boxplot(data=dataFrame, x='price')
plt.title("Box Plot of Price")
plt.xlabel('Values')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=dataFrame, x='order', y='price')
plt.title("Scatter Plot of Order V/S Price")
plt.xlabel('Order')
plt.ylabel('Price')
plt.show()

numerical_columns = ['order', 'price']
Q1 = dataFrame[numerical_columns].quantile(0.25)
Q3 = dataFrame[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

condition = (dataFrame[numerical_columns] >= lower_bound) & (dataFrame[numerical_columns] <= upper_bound)
dataFrame = dataFrame.loc[condition['order']]
dataFrame = dataFrame.loc[condition['price']]

plt.figure(figsize=(10,5))
sns.boxplot(data=dataFrame, x = 'order')
plt.title('Box Plot of Order after Outlier removal')
plt.xlabel('Values')
plt.show()

plt.figure(figsize=(14,5))
sns.boxplot(data=dataFrame, x = 'price')
plt.title('Box Plot of Price after Outlier removal')
plt.xlabel('Values')
plt.show()

dataFrame = dataFrame.reset_index(drop=True)

print("\nTransformation for the numerical features")
numer_target = dataFrame[['year', 'session ID', 'order','month', 'day', 'country', 'page 1 (main category)', 'page 2 (clothing model)', 'colour','location','model photography','page','price 2']]
numer_wo_year_df = dataFrame[['order', 'day', 'country', 'page 1 (main category)', 'page 2 (clothing model)', 'colour','location','model photography','page','price 2']]

scaler = StandardScaler()
transformed_numerica_df = scaler.fit_transform(numer_target)
transformed_num_wo_df = scaler.fit_transform(numer_wo_year_df)

y = dataFrame[['price']]
transfored_y = scaler.fit_transform(y)

##PCA for all features
pca = PCA(n_components=13)
principalComponents = pca.fit_transform(transformed_numerica_df)
print("\nThe Explained Variance Ratio is: " + str(pca.explained_variance_ratio_))

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
num_of_components = np.argmax(cumulative_explained_variance > .95) + 1
print("\nNumber of features needed to explain more than 95% of the variance: ", num_of_components)

plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.axhline(y=.95, color='r', linestyle='--')
plt.axvline(x=9, color='g', linestyle='--')
plt.show()

#### PCA without year, session id, month feature
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(transformed_num_wo_df)
print("\nThe Explained Variance Ratio is: " + str(pca.explained_variance_ratio_))

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
num_of_components = np.argmax(cumulative_explained_variance > .95) + 1
print("\nNumber of features needed to explain more than 95% of the variance: ", num_of_components)

plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.axhline(y=.95, color='r', linestyle='--')
plt.axvline(x=9, color='g', linestyle='--')
plt.show()

print("\nThe condition number for the original feature matrix:" + str(np.linalg.cond(transformed_numerica_df)))
print("\nThe condition number for the reduced feature matrix:" + str(np.linalg.cond(transformed_num_wo_df)))

svd = TruncatedSVD(n_components=13)
svd.fit(transformed_numerica_df)
reduced_transformed_numerica_df = svd.transform(transformed_numerica_df)
print("\nThe Singular Value Decomposition values are: " + str(svd.singular_values_))

## SVD without year,session id, and month
svd = TruncatedSVD(n_components=10)
svd.fit(transformed_num_wo_df)
svd.transform(transformed_num_wo_df)
print("\nThe Reduced Singular Value Decomposition values are: " + str(svd.singular_values_))

##Random Forest Analysis
rfn = RandomForestRegressor()
rfn.fit(transformed_numerica_df,transfored_y)
print("\nThe Random Forest Regressor feature importance's is " + str(rfn.feature_importances_))

feature_imp_df = pd.DataFrame({'Feature': numer_target.columns, 'Importances': rfn.feature_importances_})
feature_imp_df = feature_imp_df.sort_values(by='Importances', ascending=False)

##Plot the graph for random forest analysis
plt.figure(figsize = (10,6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importances'])
plt.xlabel('Importances')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

sel_features = feature_imp_df.loc[feature_imp_df['Importances'] > 0.001 ,'Feature'].tolist()
print("\nThe selected features are: " + str(sel_features))

##VIF for the selected features
vif_data = pd.DataFrame()
vif_data['Feature'] = numer_target.columns
vif_data['VIF'] = [variance_inflation_factor(numer_target,i) for i in range(len(numer_target.columns))]

print("\nThe VIFs are: \n" + str(vif_data))
vif_features = vif_data[vif_data['VIF'] < 5]['Feature'].tolist()
print("\nThe features selected by VIF are: \n" + str(vif_features))

##Covariance matrix
num_cov_matrix = numer_target.cov()
plt.figure(figsize = (10,6))
sns.heatmap(num_cov_matrix, annot=True)
plt.title("Covariance matrix for the Features")
plt.tight_layout()
plt.show()

##Coeeficient Correlation matrix
num_corr_matrix = numer_target.corr()
plt.figure(figsize = (10,6))
sns.heatmap(num_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Coefficients Matrix Heatmap')
plt.tight_layout()
plt.show()

##Feature Selection for Categorical target
categorical_df = dataFrame[['year','order','month','session ID', 'day', 'country', 'page 1 (main category)', 'page 2 (clothing model)', 'colour','location','model photography','page','price']]
cat_nopagetwo_df = dataFrame[['order','day','page 1 (main category)', 'page 2 (clothing model)','country','colour','location','model photography','page','price']]

scaler = StandardScaler()
transformed_cate_df = scaler.fit_transform(categorical_df)

scaler = StandardScaler()
transformed_nopagetwo_df = scaler.fit_transform(cat_nopagetwo_df)

y2 = dataFrame[['price 2']] - 1

pca = PCA(n_components=13)
pca.fit(transformed_cate_df)
pca_transformed = pca.transform(transformed_cate_df)

print("\nThe Explained variance ratio for the categorical target is: " + str(pca.explained_variance_ratio_))
cat_cum_exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)
num_cat_comp = np.argmax(cat_cum_exp_var_ratio > 0.95) + 1
print("\nNumber of features needed to explain more than 95% of the variance:", num_cat_comp)

plt.plot(range(1, len(cat_cum_exp_var_ratio) + 1),cat_cum_exp_var_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.axhline(y=.95, color='r', linestyle='--')
plt.axvline(x=10, color='g', linestyle='--')
plt.show()

##PCA with 9 components
pca = PCA(n_components=10)
pca.fit(transformed_nopagetwo_df)
pca.transform(transformed_nopagetwo_df)

print("\nThe Explained variance ratio for categorical target is: " + str(pca.explained_variance_ratio_))
cat_cum_exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)
num_cat_comp = np.argmax(cat_cum_exp_var_ratio > 0.95) + 1
print("\nNumber of features needed to explain more than 95% of the variance:", num_cat_comp)

plt.plot(range(1, len(cat_cum_exp_var_ratio) + 1),cat_cum_exp_var_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.axhline(y=.95, color='r', linestyle='--')
plt.axvline(x=9, color='g', linestyle='--')
plt.show()

print("\nThe conditional number of the features is: " + str(np.linalg.cond(transformed_cate_df)))
print("\nThe conditional number of the features is: " + str(np.linalg.cond(transformed_nopagetwo_df)))

cat_svd = TruncatedSVD(n_components=13)
cat_svd.fit(transformed_cate_df)
cat_svd.transform(transformed_cate_df)
print("\nThe SVD values of the features are: " + str(cat_svd.singular_values_))

cat_red_svd = TruncatedSVD(n_components=10)
cat_red_svd.fit(transformed_nopagetwo_df)
cat_red_svd.transform(transformed_nopagetwo_df)
print("\nThe SVD values of the features are: " + str(cat_red_svd.singular_values_))

rfc = RandomForestRegressor()
rfc.fit(transformed_cate_df, y2)

cat_feature_importances = pd.DataFrame({'Feature':categorical_df.columns, 'Importance':rfc.feature_importances_})
cat_feature_importances = cat_feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(cat_feature_importances['Feature'], cat_feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

selected_cat_features = cat_feature_importances.loc[cat_feature_importances['Importance'] > 0.01,'Feature']
selected_features_cat_list = selected_cat_features.to_list()
print("\nThe selected features are: " + str(selected_features_cat_list))

cat_vif_data =pd.DataFrame()
cat_vif_data['Feature'] = categorical_df.columns
cat_vif_data['VIF'] = [variance_inflation_factor(categorical_df, i) for i in range(len(categorical_df.columns))]
print("\nThe VIF values of the features: \n" +  str(cat_vif_data))

selected_cat_features = cat_vif_data[cat_vif_data["VIF"] < 5]["Feature"].tolist()
print("\nThe selected features are: " + str(selected_cat_features))

cat_cov_matrix = categorical_df.cov()
plt.figure(figsize=(12, 8))
sns.heatmap(cat_cov_matrix, annot=True)
plt.title('Covariance Matrix Heatmap for the features')
plt.tight_layout()
plt.show()

cat_corr_matrix = categorical_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(cat_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Coefficients Matrix Heatmap for the features')
plt.tight_layout()
plt.show()

print("\n The number of positive and negative classes are: " + str(dataFrame['price 2'].value_counts()))
dataFrame.drop(columns=['price', 'price 2'], inplace=True)
