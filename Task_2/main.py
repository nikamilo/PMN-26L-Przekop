import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def analysis():
    # pobieranie danych id=45
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # wartosci 0-4 - przerobienie na 0-1
    # 0 - brak choroby
    # 1 - choroba 
    y = (y > 0).astype(int).values.ravel()

    # brakujace dane
    print("Missing values:")
    print(X.isnull().sum())
    
    # statystyki
    print("\nStatistics:")
    print(X.describe())

    # dzialanie z brakami i danymi kategorycznymi
    # uzycie sredniej dla brakujacych wartosci
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # zbior treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # operacje z parametrem c - sila regularyzacji
    # c to odwrotnosc regularyzacji, mniejsze c oznacza silniejsza regularyzacja
    
    for c in [0.01, 1, 100]:
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(X_train, y_train)
        print(f"c = {c}, accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

    # wybor modelu
    final_model = LogisticRegression(C=1.0)
    final_model.fit(X_train, y_train)
    y_probs = final_model.predict_proba(X_test)[:, 1]

    # wizualiazcja krzywej ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.savefig('roc.png')

    # wizualizacja rozkladu zmiennych
    sns.countplot(x=y)
    plt.savefig('dist.png')
    
    print("\nReport:")
    print(classification_report(y_test, final_model.predict(X_test)))

if __name__ == "__main__":
    analysis()