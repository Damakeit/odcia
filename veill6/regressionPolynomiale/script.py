# Regression Polynomiale

# Importer les librairies
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
from sklearn.preprocessing import PolynomialFeatures



# Importer le dataset
dataset = pd.read_csv('graph.csv', sep=";")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

plt.scatter(X, y, color = 'green')


#regression lineaire simple
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y,y_pred)


# Construction du modèle de regression polynomiale
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
y_poly_pred = regressor2.predict(X_poly)
r2_poly = r2_score(y,y_poly_pred)

# Faire de nouvelles prédictions
predictions = poly_reg.fit_transform([[2019],[2025],[2040]])
regressor2.predict(predictions)


# Visualiser les résultats
plt.scatter(X, y, color = 'green')

plt.plot(X, regressor.predict(X), color = 'red')
plt.plot(X, regressor2.predict(X_poly), color = 'blue')
plt.title('Variation de température')
plt.xlabel('Année')
plt.ylabel('Variation')
plt.show()
print(r2);
print(r2_poly);





