from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#load dataset
iris = load_iris()

X = iris.data
Y = iris.target

#split data
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

#create and train model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print("model trained successfully!")
Predicitons = model.predict(X_test)
print("First Prediction:",Predicitons[0])
print("Actual Value:", Y_test[0])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Predicitons)

print("accuracy:", accuracy)