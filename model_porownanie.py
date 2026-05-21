import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

try:
    print(f"Tworzenie połączenia SQLAlchemy z Supabase ({DB_HOST})...")

    # format: postgresql+psycopg2://użytkownik:hasło@host:port/nazwa_bazy
    db_url = f"postgresql+psycopg2://laura:Laura_FitForm2026@db.ryolatpsrxuvapuaytrd.supabase.co:5432/postgres"

    engine = create_engine(db_url)

    nazwa_tabeli = "daily_logs"
    query = f"SELECT * FROM {nazwa_tabeli};"

    print("Pobieranie danych za pomocą silnika SQLAlchemy...")

    df = pd.read_sql_query(query, engine)

    print(f"Liczba pobranych rekordów: {len(df)}")
    print(df.head())

except Exception as error:
    print(f"\n Błąd połączenia: {error}")


# -------- MODELOWANIE RÓŻNICOWE------

df = df.sort_values(by=['user_id', 'data_wpisu'])

df['waga_jutro'] = df.groupby('user_id')['waga_czczo'].shift(-1)

# Y różnica wagi między dniem dzisiejszym i jutrzejszym
df['roznica_wagi'] = df['waga_jutro'] - df['waga_czczo']

# usuniecie wierszy, dla których nie jest znana różnica (ostatni dzień każdego użytkownika)
df_clean = df.dropna(subset=['roznica_wagi']).copy()

#brak wagi na czczo, wczesniej negatywnie wplynela na predykcje
X = df_clean.drop(['data_wpisu', 'waga_czczo', 'waga_jutro', 'roznica_wagi', 'id', 'user_id'], axis=1, errors='ignore')
y = df_clean['roznica_wagi']

kolumny_X = X.columns.tolist()

grupy_uzytkownikow = df_clean['user_id']
cv_grupowe = GroupKFold(n_splits=4) #GroupKFold grwanatuje że każda grupa trafia w całości albo do trenowania, albo do testowania

# ----------XGBOOST--------------
model_xgb = xgb.XGBRegressor(
    n_estimators=50,  # mało drzew, bo mała ilość danych
    max_depth=3,  # płytsze drzewo - zapobiega overfittingowi
    learning_rate=0.1,
    objective='reg:squarederror',
    reg_alpha=0.1,  # regularyzacja L1 lasso
    reg_lambda=1  # regularyzacja L2 ridge
)

"""
Zastosowano walidację krzyżową zamiast train_test_split().
Przy małym zbiorze danych klasyczny podział train_test_split jest ryzykowny,
ponieważ wynik mógłby zależeć od "szczęśliwego" lub "nieszczęśliwego" wylosowania danych do testu.
Walidacja krzyżowa (cv=5) powtarza test 5-krotnie na różnych fragmentach zbioru, co daje
stabilniejszą i bardziej obiektywną ocenę błędu RMSE.
"""


scores_xgb = cross_val_score(model_xgb, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error')

# obliczamy RMSE
rmse_scores_xgb = np.sqrt(-scores_xgb)  # zamiast błędu 0.1, zwraca -0.1, dzieki temu wie ze 0.1 wygrywa a nie 50

print("WYNIKI XGBOOST")
print(f"Błędy RMSE w poszczególnych próbach (kg): {rmse_scores_xgb}")
print(f"Średni błąd modelu (RMSE): {rmse_scores_xgb.mean():.4f} kg")

error_percentage_xgb = (rmse_scores_xgb.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd predykcji względem wagi: {error_percentage_xgb:.2f}%")

# ----------- LIGHTGBM ----------------
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(
    n_estimators=50,  # mało drzew, bo mała ilość danych
    max_depth=3,  # płytsze drzewo - zapobiega overfittingowi
    learning_rate=0.1,
    num_leaves=15,
    objective='regression',
    reg_alpha=0.1,  # regularyzacja L1 lasso
    reg_lambda=1,  # regularyzacja L2 ridge
    verbose=-1,
)

scores_lgb = cross_val_score(model_lgb, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczamy RMSE
rmse_scores_lgb = np.sqrt(-scores_lgb)  # zwraca błąd jako liczbę ujemną (np. -0.1), aby móc go "maksymalizować".
# scikit maksymalizuje, więc ważne jest aby w przypadku rmse 0.1 miało większą wartość niż 50 - stąd ta zmiana
print("\nWYNIKI LIGHTGBM")
print(f"Błędy RMSE w poszczególnych próbach (kg): {rmse_scores_lgb}")
print(f"Średni błąd modelu (RMSE): {rmse_scores_lgb.mean():.4f} kg")

error_percentage_lgb = (rmse_scores_lgb.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd predykcji względem wagi: {error_percentage_lgb:.2f}%")

# -----CATBOOST--------
from catboost import CatBoostRegressor

model_cat = CatBoostRegressor(
    iterations=200,
    depth=5,
    learning_rate=0.1,
    # cat_features= cat_features_indices, cat_features_indices=[0,3] jesli mamy dane kategoryczne podajmy ich kolumny
    loss_function='RMSE',
    verbose=0
)

# walidacja
scores_catboost = cross_val_score(model_cat, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)
# rmse
rmse_scores_cat = np.sqrt(-scores_catboost)

print("\nWYNIKI CATBOOST")
print(f"Błędy RMSE w poszczególnych próbach (kg): {rmse_scores_cat}")
print(f"Średni błąd modelu (RMSE): {rmse_scores_cat.mean():.4f} kg")

error_percentage_cat = (rmse_scores_cat.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd predykcji względem wagi: {error_percentage_cat:.2f}%")

# -----Random Forest-----

from sklearn.ensemble import RandomForestRegressor

# Inicjalizacja modelu
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,  # szybsze obliczenia
    verbose=0
)

# walidacja
scores_rf = cross_val_score(model_rf, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)
# rmse
rmse_scores_rf = np.sqrt(-scores_rf)

print("\nWYNIKI RANDOMFOREST")
print(f"Błędy RMSE w poszczególnych próbach (kg): {rmse_scores_rf}")
print(f"Średni błąd modelu (RMSE): {rmse_scores_rf.mean():.4f} kg")

error_percentage_rf = (rmse_scores_rf.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd predykcji względem wagi: {error_percentage_rf:.2f}%")

# -------REGRESJA LINIOWA---------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_lr = Pipeline([
    ('skaler', StandardScaler()),
    ('model_lr', LinearRegression())
])

scores_lr = cross_val_score(pipeline_lr, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# RMSE
rmse_scores_lr = np.sqrt(-scores_lr)

print("\nWYNIKI REGRESJI LINIOWEJ")
print(f"Błędy RMSE (kg): {rmse_scores_lr}")
print(f"Średni błąd (RMSE): {rmse_scores_lr.mean():.4f} kg")

error_percentage_lr = (rmse_scores_lr.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_lr:.2f}%")

# -------ELASTIC NET----------
from sklearn.linear_model import ElasticNet

pipeline_en = Pipeline([
    ('skaler', StandardScaler()),
    ('model_en', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
])
# im wieksza alpha tym model jest prostszy
# 0 ridge dla skolerowanych danych, gdy wszystkie cechy są ważne, 1 dla lasso gdy chcemy wyrzucić niepotrzebne cechy

# walidacja krzyżowa
scores_en = cross_val_score(pipeline_en, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczanie RMSE
rmse_scores_en = np.sqrt(-scores_en)

print("\nWYNIKI ELASTIC NET")
print(f"Błędy RMSE (kg): {rmse_scores_en}")
print(f"Średni błąd (RMSE): {rmse_scores_en.mean():.4f} kg")

error_percentage_en = (rmse_scores_en.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_en:.2f}%")

# ------Ridge Regression---------
from sklearn.linear_model import Ridge

pipeline_ridge = Pipeline([
    ('skaler', StandardScaler()),
    ('model_ridge', Ridge(alpha=1.0))
])
# walidacja krzyżowa
scores_ridge = cross_val_score(pipeline_ridge, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczanie RMSE
rmse_scores_ridge = np.sqrt(-scores_ridge)

print("\nWYNIKI RIDGE REGRESSION")
print(f"Błędy RMSE (kg): {rmse_scores_ridge}")
print(f"Średni błąd (RMSE): {rmse_scores_ridge.mean():.4f} kg")

error_percentage_ridge = (rmse_scores_ridge.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_ridge:.2f}%")

# ------SVM-----------
from sklearn.svm import SVR

pipeline_svm = Pipeline([
    ('skaler', StandardScaler()),
    ('model_svm', SVR(kernel='rbf', C=10.0, epsilon=0.1))  # rbf pozwala uczyc sie krzywych
    # C siła kary - im wyzsze tym model stara się bardziej dopasowac do każdego punktu
    # epsilon - szerokość "tuby" błędu, którą model ignoruje
])

# walidacja krzyżowa
scores_svm = cross_val_score(pipeline_svm, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczanie RMSE
rmse_scores_svm = np.sqrt(-scores_svm)

print("\nWYNIKI SVM")
print(f"Błędy RMSE (kg): {rmse_scores_svm}")
print(f"Średni błąd (RMSE): {rmse_scores_svm.mean():.4f} kg")

error_percentage_svm = (rmse_scores_svm.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_svm:.2f}%")

# ---------KNN------------
from sklearn.neighbors import KNeighborsRegressor

pipeline_knn = Pipeline([
    ('skaler', StandardScaler()),
    ('model_knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

# walidacja krzyżowa
scores_knn = cross_val_score(pipeline_knn, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczanie RMSE
rmse_scores_knn = np.sqrt(-scores_knn)

print("\nWYNIKI KNN")
print(f"Błędy RMSE (kg): {rmse_scores_knn}")
print(f"Średni błąd (RMSE): {rmse_scores_knn.mean():.4f} kg")

error_percentage_knn = (rmse_scores_knn.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_knn:.2f}%")

# -------DECISION TREE----------
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(
    max_depth=5,  # jak dużo pytań może zadać drzewo
    min_samples_split=10,  # minimalna liczba próbek, by stworzyć nowy podział
    random_state=42
)
scores_dt = cross_val_score(model_dt, X, y, cv=cv_grupowe, groups=grupy_uzytkownikow, scoring='neg_mean_squared_error', n_jobs=1)

# obliczanie RMSE
rmse_scores_dt = np.sqrt(-scores_dt)

print("\nWYNIKI DECISION TREE")
print(f"Błędy RMSE (kg): {rmse_scores_dt}")
print(f"Średni błąd (RMSE): {rmse_scores_dt.mean():.4f} kg")

error_percentage_dt = (rmse_scores_dt.mean() / df_clean['waga_czczo'].mean()) * 100
print(f"Procentowy błąd: {error_percentage_dt:.2f}%")


# -----przygtotowanie modeli do symulacji------------
model_xgb.fit(X, y, verbose=False)
model_lgb.fit(X, y)
model_cat.fit(X, y)
model_rf.fit(X, y)
pipeline_lr.fit(X, y)
pipeline_en.fit(X, y)
pipeline_ridge.fit(X, y)
pipeline_svm.fit(X, y)
pipeline_knn.fit(X, y)
model_dt.fit(X, y)


# -------------symulacja zmiany wagi--------------

def symulacja_wagi(model, dni, start_weight, kcal, bialko, spalone_kcal, cardio, silowy, kroki):
    lista_wag = [start_weight]
    obecna_waga = start_weight

    for d in range(1, dni + 1):
        dane_wiersza = {
            'zjedzone_kcal': kcal,
            'bialko_g': bialko,
            'spalone_kcal': spalone_kcal,
            'cardio_min': cardio,
            'trening_silowy': silowy,
            'kroki': kroki,
            'waga_czczo': obecna_waga,  # ta waga nadal może tu być jako punkt odniesienia
            'dzien': d
        }

        wejscie = pd.DataFrame([dane_wiersza])[kolumny_X]

        # Model zwraca małą różnicę (deltę), np. -0.15
        roznica = model.predict(wejscie)[0]

        # POPRAWKA: Dodajemy różnicę do obecnej wagi!
        obecna_waga = obecna_waga + roznica
        lista_wag.append(round(obecna_waga, 2))

    return lista_wag


# --- parametry testowe - dane z bazy są parametrami treningowymi ---
dni_prognozy = 30
waga_start = 85.0
kalk_kcal = 1600
kalk_bialko = 130
kalk_spalone = 2600
kalk_cardio = 45
kalk_silowy = 1
kalk_kroki = 14000


# --- predykcja---

wyniki_xgb = symulacja_wagi(model_xgb, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                            kalk_silowy, kalk_kroki)
wyniki_lgb = symulacja_wagi(model_lgb, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                            kalk_silowy, kalk_kroki)
wyniki_cat = symulacja_wagi(model_cat, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                            kalk_silowy, kalk_kroki)
wyniki_rf = symulacja_wagi(model_rf, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                           kalk_silowy, kalk_kroki)
wyniki_lr = symulacja_wagi(pipeline_lr, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                           kalk_silowy, kalk_kroki)
wyniki_en = symulacja_wagi(pipeline_en, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                           kalk_silowy, kalk_kroki)
wyniki_ridge = symulacja_wagi(pipeline_ridge, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone,
                              kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_svm = symulacja_wagi(pipeline_svm, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                            kalk_silowy, kalk_kroki)
wyniki_knn = symulacja_wagi(pipeline_knn, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                            kalk_silowy, kalk_kroki)
wyniki_dt = symulacja_wagi(model_dt, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio,
                           kalk_silowy, kalk_kroki)

# --- porownanie --
print("\n------POROWNANIE------")
print(f"Waga startowa: {waga_start} kg")
print(f"XGBOOST po {dni_prognozy} dniach: {wyniki_xgb[-1]:.2f} kg | Różnica: {(wyniki_xgb[-1] - waga_start):.2f} kg")
print(f"LIGHTGBM po {dni_prognozy} dniach: {wyniki_lgb[-1]:.2f} kg | Różnica: {(wyniki_lgb[-1] - waga_start):.2f} kg")
print(f"CATBOOST po {dni_prognozy} dniach: {wyniki_cat[-1]:.2f} kg | Różnica: {(wyniki_cat[-1] - waga_start):.2f} kg")
print(f"RANDOMFOREST po {dni_prognozy} dniach: {wyniki_rf[-1]:.2f} kg | Różnica: {(wyniki_rf[-1] - waga_start):.2f} kg")
print(f"LINEAR REGRESSION po {dni_prognozy} dniach: {wyniki_lr[-1]:.2f} kg | Różnica: {(wyniki_lr[-1] - waga_start):.2f} kg")
print(f"ELASTIC NET po {dni_prognozy} dniach: {wyniki_en[-1]:.2f} kg | Różnica: {(wyniki_en[-1] - waga_start):.2f} kg")
print(f"RIDGE REGRESSION po {dni_prognozy} dniach: {wyniki_ridge[-1]:.2f} kg | Różnica: {(wyniki_ridge[-1] - waga_start):.2f} kg")
print(f"SVM po {dni_prognozy} dniach: {wyniki_svm[-1]:.2f} kg | Różnica: {(wyniki_svm[-1] - waga_start):.2f} kg")
print(f"KNN po {dni_prognozy} dniach: {wyniki_knn[-1]:.2f} kg | Różnica: {(wyniki_knn[-1] - waga_start):.2f} kg")
print(f"DECISION TREE po {dni_prognozy} dniach: {wyniki_dt[-1]:.2f} kg | Różnica: {(wyniki_dt[-1] - waga_start):.2f} kg")