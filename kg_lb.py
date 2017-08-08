from sklearn import linear_model

data = [[10], [7], [6], [3], [16]]
labels = [22.05, 15.43, 13.23, 6.61, 35.27]

regr = linear_model.LinearRegression()

regr.fit(data, labels)

print(regr.predict([[15.7]]))
