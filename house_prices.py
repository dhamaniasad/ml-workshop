from sklearn import linear_model

# 1 = normaltown
# 2 = hipsterton
# 3 = skid row

data = [[3, 2000, 1], [2, 800, 2], [2, 850, 1], [1, 550, 1], [4, 2000, 3]]
labels = [250000, 300000, 150000, 78000, 150000]

regr = linear_model.LinearRegression()

regr.fit(data, labels)

print(regr.predict([[3, 2000, 2]]))
print(regr.predict([[3, 2000, 1]]))
