import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', 15)

dataFrame = pd.read_csv('e-shop-clothing.csv')

X = dataFrame[['month', 'country', 'page 1 (main category)', 'colour','location','model photography','page']]

category_mapping = {
    1: 'trousers',
    2: 'skirts',
    3: 'blouses',
    4: 'sale'
}

location_mapping = {
    1: 'top-left',
    2: 'top-in-the-middle',
    3: 'top-right',
    4: 'bottom-left',
    5: 'bottom-in-the-middle',
    6: 'bottom-right'
}

photography_mapping = {
    1: 'en-face',
    2: 'profile'
}

color_mapping = {
    1: 'beige',
    2: 'black',
    3: 'blue',
    4: 'brown',
    5: 'burgundy',
    6: 'gray',
    7: 'green',
    8: 'navy-blue',
    9: 'of-many-colors',
    10: 'olive',
    11: 'pink',
    12: 'red',
    13: 'violet',
    14: 'white'
}

month_mapping = {
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August'
}

page_mapping = {
    1: 'Page1',
    2: 'Page2',
    3: 'Page3',
    4: 'Page4',
    5: 'Page5'
}

country_mapping = {
    1: 'Australia',
    2: 'Austria',
    3: 'Belgium',
    4: 'BritishVirginIslands',
    5: 'CaymanIslands',
    6: 'ChristmasIsland',
    7: 'Croatia',
    8: 'Cyprus',
    9: 'CzechRepublic',
    10: 'Denmark',
    11: 'Estonia',
    12: 'unidentified',
    13: 'FaroeIslands',
    14: 'Finland',
    15: 'France',
    16: 'Germany',
    17: 'Greece',
    18: 'Hungary',
    19: 'Iceland',
    20: 'India',
    21: 'Ireland',
    22: 'Italy',
    23: 'Latvia',
    24: 'Lithuania',
    25: 'Luxembourg',
    26: 'Mexico',
    27: 'Netherlands',
    28: 'Norway',
    29: 'Poland',
    30: 'Portugal',
    31: 'Romania',
    32: 'Russia',
    33: 'San Marino',
    34: 'Slovakia',
    35: 'Slovenia',
    36: 'Spain',
    37: 'Sweden',
    38: 'Switzerland',
    39: 'Ukraine',
    40: 'UnitedArabEmirates',
    41: 'UnitedKingdom',
    42: 'USA',
    43: 'biz',
    44: 'com',
    45: 'int',
    46: 'net',
    47: 'org'
}

X.loc[:, 'page 1 (main category)'] = X['page 1 (main category)'].map(category_mapping)
X.loc[:, 'location'] = X['location'].map(location_mapping)
X.loc[:, 'model photography'] = X['model photography'].map(photography_mapping)
X.loc[:, 'colour'] = X['colour'].map(color_mapping)
X.loc[:, 'month'] = X['month'].map(month_mapping)
X.loc[:, 'page'] = X['page'].map(page_mapping)
X.loc[:, 'country'] = X['country'].map(country_mapping)

transactions = X.values.tolist()
trans = TransactionEncoder()
trans_array = trans.fit(transactions).transform(transactions)

encoded_df = pd.DataFrame(trans_array, columns=trans.columns_)
print(encoded_df)
print("\n")

data = apriori(encoded_df, min_support = 0.2, use_colnames = True, verbose = 1)
print(data)
print("\n")

data_ar = association_rules(data, metric = "confidence", min_threshold = 0.6)
data_ar = data_ar.sort_values(['confidence','lift'], ascending = [False, False])
print(data_ar)
print("\n")



