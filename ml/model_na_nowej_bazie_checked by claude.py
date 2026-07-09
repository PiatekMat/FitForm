import random
import pandas as pd
from faker import Faker
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# ---------config logger----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fitform_backend.log", encoding="utf-8"),  # zapis do pliku (w razie dalszych problemow)
        logging.StreamHandler()
    ]
)

load_dotenv()
db_url = os.getenv("FITFORM_DB_URL")

try:
    logging.info("Tworzenie połączenia SQLAlchemy z Supabase za pomocą FITFORM_DB_URL")

    if not db_url:
        raise ValueError("Nie znaleziono zmiennej FITFORM_DB_URL w pliku .env!")

    engine = create_engine(db_url)
    nazwa_tabeli = "daily_logs"
    query = f"SELECT * FROM {nazwa_tabeli};"

    df = pd.read_sql_query(query, engine)
    logging.info(f"Pobrano {len(df)} rekordów.")

except Exception as error:
    logging.error(f"Błąd połączenia: {error}")
    exit()

os.makedirs("static", exist_ok=True)

# -------- modelowanie roznicowe ------

df = df.sort_values(by=['user_id', 'data_wpisu'])

#uzupełniono braki w danych 'waga_czczo', aby uzytkownik z brakiem wagi_na_czczo nie został usunięty przez brak danych
df['waga_czczo'] = df.groupby('user_id')['waga_czczo'].ffill()
df['waga_czczo'] = df.groupby('user_id')['waga_czczo'].bfill()

# konwersja daty na datetime (potrzebne do cech temporalnych)
df['data_wpisu'] = pd.to_datetime(df['data_wpisu'])

# -------- FEATURE ENGINEERING --------

# bilans kaloryczny — najsilniejszy predyktor zmiany wagi
df['bilans_kcal'] = df['zjedzone_kcal'] - df['spalone_kcal']
df['bilans_pct'] = df['bilans_kcal'] / df['spalone_kcal'].replace(0, np.nan)

# cechy per kg masy ciała
df['kcal_na_kg'] = df['zjedzone_kcal'] / df['waga_czczo'].replace(0, np.nan)
df['bialko_na_kg'] = df['bialko_g'] / df['waga_czczo'].replace(0, np.nan)

# cechy interakcyjne z treningiem siłowym (binarna zmienna)
df['spalone_z_silowym'] = df['spalone_kcal'] * df['trening_silowy']
df['cardio_i_silowy'] = df['cardio_min'] * df['trening_silowy']  # combo cardio + siła
df['aktywnosc_total'] = df['cardio_min'] + df['kroki'] / 100  # ogólna aktywność

# cechy temporalne — wzorce żywieniowe różnią się w weekendy
df['dzien_tygodnia'] = df['data_wpisu'].dt.dayofweek
df['weekend'] = (df['dzien_tygodnia'] >= 5).astype(int)

# średnie kroczące per user — kontekst historyczny (trend wielodniowy)
for col in ['zjedzone_kcal', 'spalone_kcal', 'bilans_kcal']:
    df[f'{col}_ma3'] = df.groupby('user_id')[col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df[f'{col}_ma7'] = df.groupby('user_id')[col].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

logging.info("Feature engineering zakończony — dodano cechy pochodne.")

# utworzenie wagi z następnego dnia
df['waga_jutro'] = df.groupby('user_id')['waga_czczo'].shift(-1)
df['data_nastepna'] = df.groupby('user_id')['data_wpisu'].shift(-1)

df['data_nastepna'] = pd.to_datetime(df['data_nastepna'])

# faktyczna liczba dni, jaka upłynęła między wpisami
df['dni_miedzy_wpisami'] = (df['data_nastepna'] - df['data_wpisu']).dt.days

# usuniecie wierszy, dla których nie jest znana różnica (ostatni dzień każdego użytkownika)
df_clean = df.dropna(subset=['waga_jutro']).copy()

# zamiana potencjalnych nieskończoności na NaN przed obliczeniami
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Y jako różnica wagi między jutrem a dzisiaj (uśredniona dobowo), wynika to z tego że w danych rzeczywistych waga jest wpisywana co 2-3 dni a w generowamych codziennie
df_clean['roznica_wagi'] = (df_clean['waga_jutro'] - df_clean['waga_czczo']) / df_clean['dni_miedzy_wpisami']

# -------- USUNIĘCIE OUTLIERÓW W TARGECIE --------
# przycięcie ekstremalnych wartości roznica_wagi (np. błędy w danych)
Q_low = df_clean['roznica_wagi'].quantile(0.05)
Q_high = df_clean['roznica_wagi'].quantile(0.95)
n_before = len(df_clean)
df_clean = df_clean[(df_clean['roznica_wagi'] >= Q_low) & (df_clean['roznica_wagi'] <= Q_high)]
n_after = len(df_clean)
logging.info(f"Usunięto {n_before - n_after} outlierów z targetu (percentyle 5-95%). Pozostało {n_after} rekordów.")


# ------- przygotowanie i podzial danych ------

"""zastosowano bardziej skomplikowany kod ze wzgledu na to, konieczne jest uwzglednienie user_id 1-5 
zarowno w tescie jak i treningu. Są to jedyne dane od rzeczywistych osob i wazne jest aby uwglednic je w nauce modelu, 
tym samym unikajac przeuczenia czy nauki schematycznej danych wygenerowanych przy uzyciu fakera, random i profilach logicznych"""

# kolumny pomocnicze do usunięcia z X (nie są cechami wejściowymi)
kolumny_do_usuniecia = ['data_wpisu', 'waga_czczo', 'waga_jutro', 'data_nastepna',
                        'dni_miedzy_wpisami', 'roznica_wagi', 'id']

X_surowe = df_clean.drop(kolumny_do_usuniecia, axis=1, errors='ignore')
y_surowe = df_clean['roznica_wagi']

# usunięto wiersze z brakami danych (NaN) przed podziałem (tutaj zostaja usuniete wiersze z brakami innymi niz waga_czczo)
czyste_indeksy = X_surowe.dropna().index.intersection(y_surowe.dropna().index)
X_pelne = X_surowe.loc[czyste_indeksy].copy()
y_pelne = y_surowe.loc[czyste_indeksy]

# pobrano listę wszystkich unikalnych ID użytkowników w bazie
wszyscy_uzytkownicy = df_clean.loc[czyste_indeksy, 'user_id'].unique().tolist()

# filtracja użytkowników z zakresu 1-5, którzy faktycznie istnieją w bazie
kandydaci_1_5 = [uid for uid in wszyscy_uzytkownicy if uid in [1, 2, 3, 4, 5]]

# losowanie 1 uzytkownika z user_id od 1 do 5
random.seed(42)
if kandydaci_1_5:
    uzytkownik_testowy_wymuszony = random.choice(kandydaci_1_5)
    uzytkownicy_test_wymuszeni = [uzytkownik_testowy_wymuszony]
    # pozostałe osoby z zakresu 1-5 umieszczane są w zbiorze treningowym, uid-used_id
    uzytkownicy_trening_wymuszeni = [uid for uid in kandydaci_1_5 if uid != uzytkownik_testowy_wymuszony]
else:
    # jeśli w bazie nie ma osób z ID 1-5, listy zostają puste
    uzytkownicy_test_wymuszeni = []
    uzytkownicy_trening_wymuszeni = []

# reszta użytkowników z bazy, którzy nie są w zakresie id 1-5, uid-used_id
pozostali_uzytkownicy = [uid for uid in wszyscy_uzytkownicy if uid not in kandydaci_1_5]

# obliczanie ile osób łącznie powinno być w teście dla zachowania proporcji 80/20
docelowa_liczba_test = max(1, int(len(wszyscy_uzytkownicy) * 0.20))

#sprawdzenie, ilu ludzi brakuje w teście
brakujaca_liczba_test = max(0, docelowa_liczba_test - len(uzytkownicy_test_wymuszeni))

# dopełnienie zbioru testowego i treningowego pozostałymi użytkownikami
uzytkownicy_test = uzytkownicy_test_wymuszeni + pozostali_uzytkownicy[:brakujaca_liczba_test]
uzytkownicy_trening = uzytkownicy_trening_wymuszeni + pozostali_uzytkownicy[brakujaca_liczba_test:]

logging.info(f"Użytkownicy w zbiorze TRENINGOWYM (~80%): {sorted(uzytkownicy_trening)}")
logging.info(f"Użytkownicy w zbiorze TESTOWYM (~20%): {sorted(uzytkownicy_test)}")

indeksy_trening = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_trening)].index
indeksy_test = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_test)].index

kolumny_X = [col for col in X_pelne.columns if col != 'user_id']

# budowanie ostatecznych zbiorów X oraz y
X_train = X_pelne.loc[indeksy_trening, kolumny_X]
y_train = y_pelne.loc[indeksy_trening]
user_groups = df_clean.loc[indeksy_trening, 'user_id']

X_test_user = X_pelne.loc[indeksy_test, kolumny_X]
y_test_user = y_pelne.loc[indeksy_test]

logging.info(f"Cechy modelu ({len(kolumny_X)}): {kolumny_X}")
logging.info(f"Rozmiar zbioru treningowego: {len(X_train)} | testowego: {len(X_test_user)}")

# walidacja krzyżowa, GroupKFold — 5 foldów dla wiarygodniejszej walidacji
if len(uzytkownicy_trening) >= 5:
    group_cv = GroupKFold(n_splits=5)
elif len(uzytkownicy_trening) >= 3:
    group_cv = GroupKFold(n_splits=3)
else:
    from sklearn.model_selection import KFold
    group_cv = KFold(n_splits=2, shuffle=True, random_state=42)


# ----------XGBOOST--------------
model_xgb = xgb.XGBRegressor(
    n_estimators=300,  # więcej drzew z niższym learning_rate
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,  # zapobiega overfittingowi na małych grupach
    subsample=0.8,  # losowy podzbiór danych per drzewo
    colsample_bytree=0.8,  # losowy podzbiór cech per drzewo
    objective='reg:squarederror',
    reg_alpha=0.5,  # regularyzacja L1 lasso
    reg_lambda=2,  # regularyzacja L2 ridge
    early_stopping_rounds=30,  # zatrzymanie gdy model przestaje się poprawiać
)


# ----------- LIGHTGBM ----------------
model_lgb = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,  # minimum próbek w liściu
    subsample=0.8,
    colsample_bytree=0.8,
    objective='regression',
    reg_alpha=0.1,  # regularyzacja L1 lasso
    reg_lambda=2,  # regularyzacja L2 ridge
    verbose=-1,
)

# -----CATBOOST--------
model_cat = CatBoostRegressor(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=5,  # regularyzacja L2
    subsample=0.8,
    # cat_features= cat_features_indices, cat_features_indices=[0,3] jesli mamy dane kategoryczne podajmy ich kolumny
    loss_function='RMSE',
    early_stopping_rounds=30,
    verbose=0
)

# -----Random Forest-----
model_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=15,
    max_features='sqrt',  # losowy podzbiór cech — redukcja korelacji między drzewami
    random_state=42,
    n_jobs=-1,  # szybsze obliczenia
    verbose=0
)

# -------REGRESJA LINIOWA---------
pipeline_lr = Pipeline([
    ('skaler', StandardScaler()),
    ('model_lr', LinearRegression())
])

# -------ELASTIC NET----------
pipeline_en = Pipeline([
    ('skaler', StandardScaler()),
    ('model_en', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
])
# im wieksza alpha tym model jest prostszy
# 0 ridge dla skolerowanych danych, gdy wszystkie cechy są ważne, 1 dla lasso gdy chcemy wyrzucić niepotrzebne cechy

# ------Ridge Regression---------
pipeline_ridge = Pipeline([
    ('skaler', StandardScaler()),
    ('model_ridge', Ridge(alpha=1.0))
])

# ------SVM-----------
pipeline_svm = Pipeline([
    ('skaler', StandardScaler()),
    ('model_svm', SVR(kernel='rbf', C=1.0, epsilon=0.05))  # rbf pozwala uczyc sie krzywych
    # C=1.0 zamiast 10 — mniejsze C = lepsza generalizacja przy małym zbiorze
    # epsilon=0.05 — węższa tuba błędu, model jest dokładniejszy
])

# ---------KNN------------
pipeline_knn = Pipeline([
    ('skaler', StandardScaler()),
    ('model_knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

# -------DECISION TREE----------
model_dt = DecisionTreeRegressor(
    max_depth=5,  # jak dużo pytań może zadać drzewo
    min_samples_split=10,  # minimalna liczba próbek, by stworzyć nowy podział
    random_state=42
)

# -----przygtotowanie modeli do symulacji------------
logging.info("Trenowanie modeli końcowych...")

# XGBoost z early stopping — wymaga eval_set
if len(X_test_user) > 0:
    model_xgb.fit(X_train, y_train, eval_set=[(X_test_user, y_test_user)], verbose=False)
else:
    model_xgb.fit(X_train, y_train, verbose=False)

model_lgb.fit(X_train, y_train)

# CatBoost z early stopping — wymaga eval_set
if len(X_test_user) > 0:
    model_cat.fit(X_train, y_train, eval_set=(X_test_user, y_test_user))
else:
    model_cat.fit(X_train, y_train)

model_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
pipeline_en.fit(X_train, y_train)
pipeline_ridge.fit(X_train, y_train)
pipeline_svm.fit(X_train, y_train)
pipeline_knn.fit(X_train, y_train)
model_dt.fit(X_train, y_train)

# -----ENSEMBLE — ważona średnia top-3 modeli (gradient boosting)-----
model_ensemble = VotingRegressor(
    estimators=[
        ('xgb', model_xgb),
        ('lgb', model_lgb),
        ('cat', model_cat),
    ],
    weights=[0.4, 0.3, 0.3]  # dostosuj na podstawie wyników CV
)
model_ensemble.fit(X_train, y_train)
logging.info("Ensemble (XGB+LGB+CatBoost) wytrenowany.")


# -------------symulacja zmiany wagi--------------

def symulacja_wagi(model, dni, start_weight, kcal, bialko, spalone_kcal, cardio, silowy, kroki):
    lista_wag = [start_weight]
    obecna_waga = start_weight

    bilans = kcal - spalone_kcal

    for d in range(1, dni + 1):
        dane_wiersza = {
            'zjedzone_kcal': kcal,
            'bialko_g': bialko,
            'spalone_kcal': spalone_kcal,
            'cardio_min': cardio,
            'trening_silowy': silowy,
            'kroki': kroki,
            # nowe cechy pochodne
            'bilans_kcal': bilans,
            'bilans_pct': bilans / spalone_kcal if spalone_kcal > 0 else 0,
            'kcal_na_kg': kcal / obecna_waga if obecna_waga > 0 else 0,
            'bialko_na_kg': bialko / obecna_waga if obecna_waga > 0 else 0,
            'spalone_z_silowym': spalone_kcal * silowy,
            'cardio_i_silowy': cardio * silowy,
            'aktywnosc_total': cardio + kroki / 100,
            'dzien_tygodnia': d % 7,  # symulacja cykliczna
            'weekend': 1 if (d % 7) >= 5 else 0,
            # średnie kroczące — w symulacji z identycznymi danymi = wartość dzienna
            'zjedzone_kcal_ma3': kcal,
            'zjedzone_kcal_ma7': kcal,
            'spalone_kcal_ma3': spalone_kcal,
            'spalone_kcal_ma7': spalone_kcal,
            'bilans_kcal_ma3': bilans,
            'bilans_kcal_ma7': bilans,
        }

        wejscie = pd.DataFrame([dane_wiersza])[kolumny_X]
        roznica = model.predict(wejscie)[0]
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
wyniki_ensemble = symulacja_wagi(model_ensemble, dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone,
                                 kalk_cardio, kalk_silowy, kalk_kroki)

# --- porownanie --
logging.info("------ PORÓWNANIE KOŃCOWE ------")
logging.info(f"Waga startowa: {waga_start:.2f} kg")
logging.info(
    f"XGBOOST po {dni_prognozy} dniach: {wyniki_xgb[-1]:.2f} kg | Różnica: {(wyniki_xgb[-1] - waga_start):.2f} kg")
logging.info(
    f"LIGHTGBM po {dni_prognozy} dniach: {wyniki_lgb[-1]:.2f} kg | Różnica: {(wyniki_lgb[-1] - waga_start):.2f} kg")
logging.info(
    f"CATBOOST po {dni_prognozy} dniach: {wyniki_cat[-1]:.2f} kg | Różnica: {(wyniki_cat[-1] - waga_start):.2f} kg")
logging.info(
    f"RANDOMFOREST po {dni_prognozy} dniach: {wyniki_rf[-1]:.2f} kg | Różnica: {(wyniki_rf[-1] - waga_start):.2f} kg")
logging.info(
    f"LINEAR REGRESSION po {dni_prognozy} dniach: {wyniki_lr[-1]:.2f} kg | Różnica: {(wyniki_lr[-1] - waga_start):.2f} kg")
logging.info(
    f"ELASTIC NET po {dni_prognozy} dniach: {wyniki_en[-1]:.2f} kg | Różnica: {(wyniki_en[-1] - waga_start):.2f} kg")
logging.info(
    f"RIDGE REGRESSION po {dni_prognozy} dniach: {wyniki_ridge[-1]:.2f} kg | Różnica: {(wyniki_ridge[-1] - waga_start):.2f} kg")
logging.info(f"SVM po {dni_prognozy} dniach: {wyniki_svm[-1]:.2f} kg | Różnica: {(wyniki_svm[-1] - waga_start):.2f} kg")
logging.info(f"KNN po {dni_prognozy} dniach: {wyniki_knn[-1]:.2f} kg | Różnica: {(wyniki_knn[-1] - waga_start):.2f} kg")
logging.info(
    f"DECISION TREE po {dni_prognozy} dniach: {wyniki_dt[-1]:.2f} kg | Różnica: {(wyniki_dt[-1] - waga_start):.2f} kg")
logging.info(
    f"ENSEMBLE po {dni_prognozy} dniach: {wyniki_ensemble[-1]:.2f} kg | Różnica: {(wyniki_ensemble[-1] - waga_start):.2f} kg")

# ----------------WYKRES PORÓWNAWCZY-----------------

try:
    if len(X_test_user) > 0:
        logging.info("Generowanie wykresu dopasowania dla wszystkich 11 modeli...")

        # wybor użytkownika z największą liczbą wpisów w bazie
        najczestszy_user = df_clean['user_id'].value_counts().index[0]
        df_user = df_clean[df_clean['user_id'] == najczestszy_user].sort_values(by='data_wpisu').copy()

        if len(df_user) > 1:
            X_user = df_user[kolumny_X]
            delta = df_user['roznica_wagi'].values
            waga_start_user = df_user['waga_czczo'].iloc[0]

            historie_wag = {
                'Rzeczywista': [waga_start_user],
                'XGBoost': [waga_start_user],
                'LightGBM': [waga_start_user],
                'CatBoost': [waga_start_user],
                'Random Forest': [waga_start_user],
                'Linear Regression': [waga_start_user],
                'Elastic Net': [waga_start_user],
                'Ridge Regression': [waga_start_user],
                'SVM (RBF)': [waga_start_user],
                'KNN': [waga_start_user],
                'Decision Tree': [waga_start_user],
                'Ensemble': [waga_start_user]
            }

            # predykcje delty dla wszystkich modeli
            delty_modele = {
                'XGBoost': model_xgb.predict(X_user),
                'LightGBM': model_lgb.predict(X_user),
                'CatBoost': model_cat.predict(X_user),
                'Random Forest': model_rf.predict(X_user),
                'Linear Regression': pipeline_lr.predict(X_user),
                'Elastic Net': pipeline_en.predict(X_user),
                'Ridge Regression': pipeline_ridge.predict(X_user),
                'SVM (RBF)': pipeline_svm.predict(X_user),
                'KNN': pipeline_knn.predict(X_user),
                'Decision Tree': model_dt.predict(X_user),
                'Ensemble': model_ensemble.predict(X_user)
            }

            for i in range(len(df_user)):
                # prawdziwa waga
                historie_wag['Rzeczywista'].append(historie_wag['Rzeczywista'][-1] + delta[i])

                # wagi z modeli
                for nazwa_modelu, predykcje_delty in delty_modele.items():
                    historie_wag[nazwa_modelu].append(historie_wag[nazwa_modelu][-1] + predykcje_delty[i])

            # oś X
            dni_user = list(range(0, len(historie_wag['Rzeczywista'])))

            plt.figure(figsize=(14, 8))

            plt.plot(dni_user, historie_wag['Rzeczywista'], label='WARTOŚĆ RZECZYWISTA',
                     color='black', linewidth=3.5, marker='o', zorder=5)

            # Style linii dla modeli, żeby wykres był czytelny
            plt.plot(dni_user, historie_wag['XGBoost'], label='XGBoost', linewidth=1.8, linestyle='-')
            plt.plot(dni_user, historie_wag['LightGBM'], label='LightGBM', linewidth=1.8, linestyle='-')
            plt.plot(dni_user, historie_wag['CatBoost'], label='CatBoost', linewidth=1.8, linestyle='-')
            plt.plot(dni_user, historie_wag['Random Forest'], label='Random Forest', linewidth=1.5, linestyle='--')
            plt.plot(dni_user, historie_wag['Linear Regression'], label='Linear Regression', linewidth=1.5, linestyle=':')
            plt.plot(dni_user, historie_wag['Elastic Net'], label='Elastic Net', linewidth=1.5, linestyle=':')
            plt.plot(dni_user, historie_wag['Ridge Regression'], label='Ridge Regression', linewidth=1.5, linestyle=':')
            plt.plot(dni_user, historie_wag['SVM (RBF)'], label='SVM (RBF)', linewidth=1.5, linestyle='-.')
            plt.plot(dni_user, historie_wag['KNN'], label='KNN', linewidth=1.5, linestyle='-.')
            plt.plot(dni_user, historie_wag['Decision Tree'], label='Decision Tree', linewidth=1.5, linestyle='--')
            plt.plot(dni_user, historie_wag['Ensemble'], label='ENSEMBLE (XGB+LGB+Cat)',
                     color='red', linewidth=2.5, linestyle='-', marker='s', markersize=4, zorder=4)

            plt.title(
                f"Analiza porównawcza modeli na użytkowniku (ID: {najczestszy_user})",
                fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Dni", fontsize=11)
            plt.ylabel("Masa ciała (kg)", fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

            # zapis do pliku gdzie bedzie przechowywany nasz wykres
            sciezka = "static/wykres.png"
            plt.tight_layout()
            plt.savefig(sciezka, dpi=150, bbox_inches='tight')
            plt.close()

            logging.info(f"Zapisanie wykresu w: {sciezka}")
        else:
            logging.warning("Za mało danych, aby wygenerować pełną linię czasu.")
    else:
        logging.warning("Brak danych testowych. Pomijam wykres czasu.")

except Exception as e:
    logging.error(f"Błąd podczas tworzenia wykresu {e}")

# ------TABELA PORÓWNAWCZA------

try:
    logging.info("Generowanie tabeli...")

    # metryki
    scoring_metrics = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    lista_modeli = [
        ('XGBoost', model_xgb), ('LightGBM', model_lgb), ('CatBoost', model_cat),
        ('Random Forest', model_rf), ('Linear Regression', pipeline_lr),
        ('Elastic Net', pipeline_en), ('Ridge Regression', pipeline_ridge),
        ('SVM (RBF)', pipeline_svm), ('KNN', pipeline_knn), ('Decision Tree', model_dt),
        ('Ensemble', model_ensemble)
    ]

    wiersze = []

    """
    Zastosowano walidację krzyżową GroupKFold (odmianę K-Fold Cross-Validation).
    Klasyczny podział na dwie części byłby zbyt losowy i ryzykowny dla małego zbioru danych. 

    Algorytm dzieli bazę na 4 części (foldy) według 'user_id'. Cała historia jednego użytkownika 
    trafia w całości albo do treningu, albo do testu. Model jest więc sprawdzany na osobach, 
    których wcześniej "nie widział". Średnia z 4 prób daje obiektywny błąd RMSE.
    """

    for nazwa, model in lista_modeli:
        cv = cross_validate(model, X_train, y_train, cv=group_cv, groups=user_groups, scoring=scoring_metrics, n_jobs=1)

        rmse = -cv['test_rmse']
        mae = -cv['test_mae']
        r2 = cv['test_r2']

        # obliczenia na wydzielonym zbiorze testowym dla konkretnego użytkownika
        if len(X_test_user) > 0:
            predykcje_testowe = model.predict(X_test_user)
            test_rmse = np.sqrt(np.mean((y_test_user - predykcje_testowe) ** 2))
            test_mae = np.mean(np.abs(y_test_user - predykcje_testowe))

            # obliczanie R2 dla testu
            wariancja_calkowita = np.sum((y_test_user - np.mean(y_test_user)) ** 2)
            if wariancja_calkowita > 0:
                test_r2 = 1 - (np.sum((y_test_user - predykcje_testowe) ** 2) / wariancja_calkowita)
            else:
                test_r2 = 0.0
        else:
            test_rmse, test_mae, test_r2 = np.nan, np.nan, np.nan

        wiersze.append({
            'Model': nazwa,
            'RMSE mean CV [kg]': round(np.mean(rmse), 4),
            'RMSE std CV [kg]': round(np.std(rmse), 4),
            'RMSE pct of range CV [kg]': round(np.max(rmse) - np.min(rmse), 4),
            'MAE mean CV [kg]': round(np.mean(mae), 4),
            'MAE std CV [kg]': round(np.std(mae), 4),
            'R2 mean CV [kg]': round(np.mean(r2), 4),
            'R2 std CV [kg]': round(np.std(r2), 4),
            'Test RMSE [kg]': round(test_rmse, 4) if not np.isnan(test_rmse) else "Brak danych",
            'Test MAE [kg]': round(test_mae, 4) if not np.isnan(test_mae) else "Brak danych",
            'Test R2 [kg]': round(test_r2, 4) if not np.isnan(test_r2) else "Brak danych"
        })

    df_wyniki = pd.DataFrame(wiersze)

    # sortowanie tabeli
    df_wyniki = df_wyniki.sort_values(by='MAE mean CV [kg]', ascending=True).reset_index(drop=True)

    logging.info(
        "Tabela porównawcza")
    logging.info(f"\n{df_wyniki.to_string(index=False)}")

    sciezka_csv = "static/tabela_porownawcza.csv"
    df_wyniki.to_csv(sciezka_csv, index=False, encoding='utf-8-sig')
    logging.info(f"Zapisano tabelę w: {sciezka_csv}")

except Exception as e:
    logging.error(f"Błąd podczas generowania tabeli: {e}")