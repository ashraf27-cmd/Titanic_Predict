import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


cl = pd.read_csv("Titanic.csv")

print("Information Regarding Dataset: ")
print(cl.info())

print("First Few Rows From Dataset: ")
print(cl.head())

print("Null Values:\n",cl.isnull().sum())

print("Duplicate Values: ",cl.duplicated().sum())

cl = cl.drop_duplicates()
cl = cl.drop('Cabin', axis=1)

cl['Age'] = cl['Age'].fillna(cl['Age'].median())

cl['Embarked'] = cl['Embarked'].fillna(cl['Embarked'].mode()[0])
print("Null Values:\n",cl.isnull().sum())


#EDA

# Survival By Gender
sns.countplot(x='Sex', hue='Survived', data=cl)
plt.title("Survival by Gender")
plt.show()

# Survival Number
sns.countplot(x='Survived', data=cl)
plt.title("Survival Distribution")
plt.show()

# Age Distribution

sns.histplot(cl['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Survival By Passenger
sns.countplot(x='Pclass', hue='Survived', data=cl)
plt.title("Survival by Passenger Class")
plt.show()


cl['Sex'] = cl['Sex'].map({'male':0,'female':1})
cl.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
print(cl.head())

cl['Embarked'] = cl['Embarked'].map({'S': 0,'C': 1,'Q': 2})

print(cl.head())

corr1 = cl.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr1, annot=True, cmap='coolwarm')
plt.title("Correlation")
plt.show()

# ML Method
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X = cl.drop('Survived', axis=1)
y = cl['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree Results")
print(classification_report(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("Accuracy:", accuracy_score(y_test, dt_pred))


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100,random_state=42)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Results")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))


# Final Comparison
log_acc = accuracy_score(y_test, pred)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

comparison = pd.DataFrame({'Model': ['Logistic Regression','Decision Tree','Random Forest'],'Accuracy': [log_acc,dt_acc,rf_acc]})

print("\nModel Comparison")
print(comparison)