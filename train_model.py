import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from model.preprocess import preprocess

df = pd.read_csv('data/Titanic-Dataset.csv')
df = preprocess(df)
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, 'model/model.pkl')
print("Model trained and saved to model/model.pkl")