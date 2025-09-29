#Train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_sca, y, test_size=0.2, random_state=42)  # shuffle=False for time-series

#Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

#Predicting
y_pred = model.predict(x_test)

#Train RandomForest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Predict
y_pred1 = rf_model.predict(x_test)