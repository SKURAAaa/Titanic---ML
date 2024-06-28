# Krok 1: Importowanie bibliotek
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# plt.show()
# %matplotlib inline

# Krok 2: Wczytanie zestawów danych
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Krok 3: Eksploracyjna analiza danych (EDA)
print(train_data.head())
print(train_data.describe())
print(train_data.info())

# Wizualizacja wskaźnika przeżywalności według płci i klasy pasażerskiej
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.show()
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()

# Krok 4: Przetwarzanie danych (uzupełnianie brakujących wartości, konwersja danych kategorycznych)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])

# Przeskalowanie cech wiek i opłata
scaler = StandardScaler()
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])

# Uzupełnienie brakujących wartości w kolumnie 'Fare' w obu zestawach danych
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Krok 5: Budowanie modelu uczenia maszynowego
X_train = train_data[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y_train = train_data['Survived']
model = LogisticRegression()
model.fit(X_train, y_train)

# Krok 6: Ocena modelu za pomocą walidacji krzyżowej
scores = cross_val_score(model, X_train, y_train, cv=5)
print('Wyniki walidacji krzyżowej:', scores)
print('Średni wynik:', np.mean(scores))

# Krok 7: Ulepszanie modelu za pomocą klasyfikatora drzew decyzyjnych i Grid Search
params = {'max_depth': [3, 5, 7, 9], 'min_samples_split': [2, 5, 10, 20]}
tree = DecisionTreeClassifier()
clf = GridSearchCV(tree, params, cv=5)
clf.fit(X_train, y_train)
print('Najlepsze parametry:', clf.best_params_)
print('Najlepszy wynik:', clf.best_score_)

# Krok 8: Dokonanie prognoz za pomocą zoptymalizowanego klasyfikatora drzew decyzyjnych
X_test = test_data[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
predictions = clf.predict(X_test)

# Krok 9: Tworzenie pliku do zgłoszenia i zapisywanie go jako CSV
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)

# Importowanie bibliotek dla RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Wczytywanie danych
train_df = pd.read_csv('train.csv')

# Podział na cechy i zmienną docelową
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

# Definiowanie pipeline
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier())
])

# Definiowanie siatki parametrów dla GridSearchCV
param_grid = {
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5, 10, 20]
}

# Wykonywanie GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Drukowanie wyników
print("Wyniki walidacji krzyżowej:", grid_search.cv_results_['mean_test_score'])
print("Średni wynik:", grid_search.best_score_)
print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepszy wynik:", grid_search.best_score_)

# Wczytywanie danych testowych
test_df = pd.read_csv('test.csv')

# Uzupełnianie brakujących wartości w danych testowych
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# Dokonywanie prognoz na danych testowych
X_test = test_df.copy()
predictions = grid_search.predict(X_test)

# Zapisywanie prognoz do pliku CSV
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
submission_df.to_csv('submission.csv', index=False)
