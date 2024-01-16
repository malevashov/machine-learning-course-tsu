import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CreditCardClients.csv')
data = df.values
row, col = data.shape
X = data[:, 0:col-1]
y_target = data[:, col-1]
y = LabelEncoder().fit_transform(y_target)
print('\nLoaded Pandas dataframe\n------------------------------\n', df.head())
print('\nExtract data (first 5) \n------------------------------\n', data[0:col])
print('\nX (first 5) \n------------------------------\n', X[0:5])
print('\ny (first 5) \n------------------------------\n', y[0:5])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

logreg = LogisticRegression(solver='liblinear', C=0.5, multi_class='ovr', random_state=0)
logreg.fit(x_train, y_train)

print('\nmodel parameter coef\n------------------------------\n', np.around(logreg.coef_, decimals=3))
print('\nmodel parameter intercept\n------------------------------\n', np.around(logreg.intercept_, decimals=3))

d1 = np.array([[29997, 150000, 1, 3, 2, 43, -1, -1, -1, -1, 0, 0, 1683, 1828, 3502, 8979, 5190, 0, 1837, 3526, 8998,
                129, 0, 0], [29998, 30000, 1, 2, 2, 37, 4, 3, 2, -1, 0, 0, 3565, 3356, 2758, 20878, 20582, 19357, 0, 0,
                             22000, 4200, 2000, 3100]])
d1_std = scaler.transform(d1)
d1_pred = logreg.predict(d1_std)
print('\nprediction of input data: [[29997, 150000, 1, 3, 2, 43, -1, -1, -1, -1, 0, 0, 1683, 1828, 3502, 8979, 5190, 0, 1837, 3526, 8998,129, 0, 0], [29998, 30000, 1, 2, 2, 37, 4, 3, 2, -1, 0, 0, 3565, 3356, 2758, 20878, 20582, 19357, 0, 0, 22000, 4200, 2000, 3100]]\n------------------------------\n',
      d1_pred[:])

x_test = scaler.transform(x_test)
y_pred = logreg.predict(x_test)

s1 = logreg.score(x_train, y_train)
s2 = logreg.score(x_test, y_test)

print('\nscore of training and test data\n-------------------\n', np.around(s1, decimals=3), np.around(s2, decimals=3))



