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

#konwersja daty na datetime została przeniesiona wyżej, aby umożliwić poprawne sortowanie i inżynierię cech
df['data_wpisu'] = pd.to_datetime(df['data_wpisu'])

# -------- feature engineering --------
# bilans kaloryczny
df['bilans_kcal'] = df['zjedzone_kcal'] - df['spalone_kcal']
df['bilans_pct'] = df['bilans_kcal'] / df['spalone_kcal'].replace(0, np.nan)

# cechy per kg masy ciała
df['kcal_na_kg'] = df['zjedzone_kcal'] / df['waga_czczo'].replace(0, np.nan)
df['bialko_na_kg'] = df['bialko_g'] / df['waga_czczo'].replace(0, np.nan)

# cechy związane z aktywnością
df['spalone_z_silowym'] = df['spalone_kcal'] * df['trening_silowy']
df['cardio_i_silowy'] = df['cardio_min'] * df['trening_silowy']  # cardio + siła
df['aktywnosc_total'] = df['cardio_min'] + df['kroki'] / 100  # ogólna aktywność

# wzorce żywieniowe różnią się w weekendy
df['dzien_tygodnia'] = df['data_wpisu'].dt.dayofweek
df['weekend'] = (df['dzien_tygodnia'] >= 5).astype(int)

# średnie kroczące per user — kontekst historyczny (trend wielodniowy)
for col in ['zjedzone_kcal', 'spalone_kcal', 'bilans_kcal']:
    df[f'{col}_ma3'] = df.groupby('user_id')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df[f'{col}_ma7'] = df.groupby('user_id')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())

logging.info("Dodano cechy kontekstowe.")

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

# -------- usunięcie outlierów --------
# dodano przycinanie skrajnych 5% wartości (percentyle 5-95%), aby usunąć szum i błędy wpisów z bazy danych
Q_low = df_clean['roznica_wagi'].quantile(0.05)
Q_high = df_clean['roznica_wagi'].quantile(0.95)
df_clean = df_clean[(df_clean['roznica_wagi'] >= Q_low) & (df_clean['roznica_wagi'] <= Q_high)]


# ------- przygotowanie i podzial danych ------

"""zastosowano bardziej skomplikowany kod ze wzgledu na to, konieczne jest uwzglednienie user_id 1-5 
zarowno w tescie jak i treningu. Są to jedyne dane od rzeczywistych osob i wazne jest aby uwglednic je w nauce modelu, 
tym samym unikajac przeuczenia czy nauki schematycznej danych wygenerowanych przy uzyciu fakera, random i profilach logicznych"""


kolumny_do_usuniecia = ['data_wpisu', 'waga_czczo', 'waga_jutro', 'data_nastepna', 'dni_miedzy_wpisami', 'roznica_wagi', 'id']
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

logging.info(f"Użytkownicy w zbiorze treningowym (80%): {sorted(uzytkownicy_trening)}")
logging.info(f"Użytkownicy w zbiorze testowym (20%): {sorted(uzytkownicy_test)}")

indeksy_trening = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_trening)].index
indeksy_test = df_clean.loc[czyste_indeksy][df_clean.loc[czyste_indeksy, 'user_id'].isin(uzytkownicy_test)].index

kolumny_X = [col for col in X_pelne.columns if col != 'user_id']

# wprowadzono rzutowanie typów zmiennych na float32. Zapobiega to błędom przy LightGBM i XGBoost wynikającym z niekompatybilności typów logicznych int/float bazy danych.
X_train = X_pelne.loc[indeksy_trening, kolumny_X].astype(np.float32)
y_train = y_pelne.loc[indeksy_trening].astype(np.float32)
user_groups = df_clean.loc[indeksy_trening, 'user_id']

X_test_user = X_pelne.loc[indeksy_test, kolumny_X].astype(np.float32)
y_test_user = y_pelne.loc[indeksy_test].astype(np.float32)

# walidacja krzyżowa, kfold to wymuszenie 'tasowania' gdy uzytkownikow jest mniej niz 3 i nie można ich podzielić w grupy
if len(uzytkownicy_trening) >= 3:
    group_cv = GroupKFold(n_splits=3)
else:
    from sklearn.model_selection import KFold
    group_cv = KFold(n_splits=2, shuffle=True, random_state=42)


# --- przygotowanie modeli (zapobiega data leakingowi  ---
def przygotowanie_modeli():
    return {
        # ----------XGBOOST--------------
        'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, objective='reg:squarederror',
                                    reg_alpha=0.3, reg_lambda=2.0, random_state=42),

        # ----------- LIGHTGBM ----------------
        'LightGBM': lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, num_leaves=15,
                                      objective='regression', reg_alpha=0.3, reg_lambda=2.0, verbose=-1, random_state=42),

        # -----CATBOOST--------
        'CatBoost': CatBoostRegressor(iterations=350, depth=4, learning_rate=0.05, loss_function='RMSE', verbose=0, random_state=42),


        # -----Random Forest-----
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),


        # -------REGRESJA LINIOWA---------
        'Linear Regression': Pipeline([('skaler', StandardScaler()), ('model_lr', LinearRegression())]),


        # -------ELASTIC NET----------
        # im wieksza alpha tym model jest prostszy
        # 0 ridge dla skolerowanych danych, gdy wszystkie cechy są ważne, 1 dla lasso gdy chcemy wyrzucić niepotrzebne cechy
        'Elastic Net': Pipeline([('skaler', StandardScaler()), ('model_en', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))]),


        # ------Ridge Regression---------
        'Ridge Regression': Pipeline([('skaler', StandardScaler()), ('model_ridge', Ridge(alpha=1.0))]),


        # ------SVM-----------
        # rbf pozwala uczyc sie krzywych
        # C siła kary - im wyzsze tym model stara się bardziej dopasowac do każdego punktu
        # epsilon - szerokość "tuby" błędu, którą model ignoruje
        'SVM (RBF)': Pipeline([('skaler', StandardScaler()), ('model_svm', SVR(kernel='rbf', C=5.0, epsilon=0.05))]),


        # ---------KNN------------
        'KNN': Pipeline([('skaler', StandardScaler()), ('model_knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),


       # -------DECISION TREE----------
        # jak dużo pytań może zadać drzewo
        # minimalna liczba próbek, by stworzyć nowy podział
        'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    }


# -----przygotowanie modeli do symulacji v2------------
logging.info("Trenowanie modeli końcowych...")
modele_wytrenowane = przygotowanie_modeli()
for nazwa, model in modele_wytrenowane.items():
    model.fit(X_train, y_train)

# -----ensemble - ważona średnia modeli gradient boosting -----
# wprowadzono w pełni poprawny obiekt VotingRegressor uczący się na niezależnych, zoptymalizowanych boosterach
model_ensemble = VotingRegressor(
    estimators=[
        ('xgb', xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42).fit(X_train, y_train)),
        ('lgb', lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, verbose=-1, random_state=42).fit(X_train, y_train)),
        ('cat', CatBoostRegressor(iterations=350, depth=4, verbose=0, random_state=42).fit(X_train, y_train))
    ],
    weights=[0.4, 0.3, 0.3]
).fit(X_train, y_train)
modele_wytrenowane['Ensemble'] = model_ensemble


# -------------symulacja zmiany wagi--------------

def symulacja_wagi(model, dni, start_weight, kcal, bialko, spalone_kcal, cardio, silowy, kroki):
    lista_wag = [start_weight]
    obecna_waga = start_weight
    bilans = kcal - spalone_kcal

    for d in range(1, dni + 1):
        # rozszerzono strukturę słownika symulacji o nowo dodane zaawansowane cechy kontekstowe i średnie kroczące
        dane_wiersza = {
            'zjedzone_kcal': kcal, 'bialko_g': bialko, 'spalone_kcal': spalone_kcal,
            'cardio_min': cardio, 'trening_silowy': silowy, 'kroki': kroki,
            'bilans_kcal': bilans, 'bilans_pct': bilans / spalone_kcal if spalone_kcal > 0 else 0,
            'kcal_na_kg': kcal / obecna_waga if obecna_waga > 0 else 0,
            'bialko_na_kg': bialko / obecna_waga if obecna_waga > 0 else 0,
            'spalone_z_silowym': spalone_kcal * silowy, 'cardio_i_silowy': cardio * silowy,
            'aktywnosc_total': cardio + kroki / 100, 'dzien_tygodnia': d % 7, 'weekend': 1 if (d % 7) >= 5 else 0,
            'zjedzone_kcal_ma3': kcal, 'zjedzone_kcal_ma7': kcal, 'spalone_kcal_ma3': spalone_kcal,
            'spalone_kcal_ma7': spalone_kcal, 'bilans_kcal_ma3': bilans, 'bilans_kcal_ma7': bilans,
        }

        wejscie = pd.DataFrame([dane_wiersza])[kolumny_X].astype(np.float32)
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
# MODYFIKACJA: Dodano wywołanie symulacji dla modelu Ensemble
wyniki_xgb = symulacja_wagi(modele_wytrenowane['XGBoost'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_lgb = symulacja_wagi(modele_wytrenowane['LightGBM'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_cat = symulacja_wagi(modele_wytrenowane['CatBoost'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_rf = symulacja_wagi(modele_wytrenowane['Random Forest'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_lr = symulacja_wagi(modele_wytrenowane['Linear Regression'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_en = symulacja_wagi(modele_wytrenowane['Elastic Net'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_ridge = symulacja_wagi(modele_wytrenowane['Ridge Regression'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_svm = symulacja_wagi(modele_wytrenowane['SVM (RBF)'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_knn = symulacja_wagi(modele_wytrenowane['KNN'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_dt = symulacja_wagi(modele_wytrenowane['Decision Tree'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)
wyniki_ensemble = symulacja_wagi(modele_wytrenowane['Ensemble'], dni_prognozy, waga_start, kalk_kcal, kalk_bialko, kalk_spalone, kalk_cardio, kalk_silowy, kalk_kroki)

# --- porownanie --
logging.info("------ PORÓWNANIE KOŃCOWE ------")
logging.info(f"Waga startowa: {waga_start:.2f} kg")
logging.info(f"XGBOOST po {dni_prognozy} dniach: {wyniki_xgb[-1]:.2f} kg | Różnica: {(wyniki_xgb[-1] - waga_start):.2f} kg")
logging.info(f"LIGHTGBM po {dni_prognozy} dniach: {wyniki_lgb[-1]:.2f} kg | Różnica: {(wyniki_lgb[-1] - waga_start):.2f} kg")
logging.info(f"CATBOOST po {dni_prognozy} dniach: {wyniki_cat[-1]:.2f} kg | Różnica: {(wyniki_cat[-1] - waga_start):.2f} kg")
logging.info(f"RANDOMFOREST po {dni_prognozy} dniach: {wyniki_rf[-1]:.2f} kg | Różnica: {(wyniki_rf[-1] - waga_start):.2f} kg")
logging.info(f"LINEAR REGRESSION po {dni_prognozy} dniach: {wyniki_lr[-1]:.2f} kg | Różnica: {(wyniki_lr[-1] - waga_start):.2f} kg")
logging.info(f"ELASTIC NET po {dni_prognozy} dniach: {wyniki_en[-1]:.2f} kg | Różnica: {(wyniki_en[-1] - waga_start):.2f} kg")
logging.info(f"RIDGE REGRESSION po {dni_prognozy} dniach: {wyniki_ridge[-1]:.2f} kg | Różnica: {(wyniki_ridge[-1] - waga_start):.2f} kg")
logging.info(f"SVM po {dni_prognozy} dniach: {wyniki_svm[-1]:.2f} kg | Różnica: {(wyniki_svm[-1] - waga_start):.2f} kg")
logging.info(f"KNN po {dni_prognozy} dniach: {wyniki_knn[-1]:.2f} kg | Różnica: {(wyniki_knn[-1] - waga_start):.2f} kg")
logging.info(f"DECISION TREE po {dni_prognozy} dniach: {wyniki_dt[-1]:.2f} kg | Różnica: {(wyniki_dt[-1] - waga_start):.2f} kg")
logging.info(f"ENSEMBLE po {dni_prognozy} dniach: {wyniki_ensemble[-1]:.2f} kg | Różnica: {(wyniki_ensemble[-1] - waga_start):.2f} kg")


# ------ tabela porównawcza------

try:
    logging.info("Generowanie tabeli...")

    # metryki
    scoring_metrics = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    # utworzono oddzielny zestaw modeli do walidacji krzyżowej (CV) chroniący przed przeciekiem danych
    modele_do_cv = przygotowanie_modeli()
    modele_do_cv['Ensemble'] = VotingRegressor(
        estimators=[('xgb', przygotowanie_modeli()['XGBoost']), ('lgb', przygotowanie_modeli()['LightGBM']), ('cat', przygotowanie_modeli()['CatBoost'])],
        weights=[0.4, 0.3, 0.3]
    )

    wiersze = []

    """
    Zastosowano walidację krzyżową GroupKFold (odmianę K-Fold Cross-Validation).
    Klasyczny podział na dwie części byłby zbyt losowy i ryzykowny dla małego zbioru danych. 

    Algorytm dzieli bazę na 4 części (foldy) według 'user_id'. Cała historia jednego użytkownika 
    trafia w całości albo do treningu, albo do testu. Model jest więc sprawdzany na osobach, 
    których wcześniej "nie widział". Średnia z 4 prób daje obiektywny błąd RMSE.
    """

    for nazwa, model_cv in modele_do_cv.items():
        cv = cross_validate(model_cv, X_train, y_train, cv=group_cv, groups=user_groups, scoring=scoring_metrics, n_jobs=1)

        rmse = -cv['test_rmse']
        mae = -cv['test_mae']
        r2 = cv['test_r2']

        # obliczenia na wydzielonym zbiorze testowym dla konkretnego użytkownika
        # ewaluacja zbioru testowego odwołuje się bezpośrednio do 'modele_wytrenowane'
        m_wytrenowany = modele_wytrenowane[nazwa]
        if len(X_test_user) > 0:
            predykcje_testowe = m_wytrenowany.predict(X_test_user)
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
            'RMSE mean CV': round(np.mean(rmse), 4),
            'RMSE std CV': round(np.std(rmse), 4),
            'MAE mean CV': round(np.mean(mae), 4),
            'R2 mean CV': round(np.mean(r2), 4),
            'Test RMSE': round(test_rmse, 4) if not np.isnan(test_rmse) else "Brak danych",
            'Test MAE': round(test_mae, 4) if not np.isnan(test_mae) else "Brak danych",
            'Test R2': round(test_r2, 4) if not np.isnan(test_r2) else "Brak danych"
        })

    df_wyniki = pd.DataFrame(wiersze)

    # sortowanie tabeli
    df_wyniki = df_wyniki.sort_values(by='MAE mean CV', ascending=True).reset_index(drop=True)

    logging.info("Tabela porównawcza gotowa.")
    print("\n TABELA WYNIKÓW ")
    print(df_wyniki.to_string(index=False))

    sciezka_csv = "static/tabela_porownawcza.csv"
    df_wyniki.to_csv(sciezka_csv, index=False, encoding='utf-8-sig')
    logging.info(f"Zapisano tabelę w: {sciezka_csv}")

except Exception as e:
    logging.error(f"Błąd podczas generowania tabeli: {e}")


# eksport modelu lightgbm i ensemble do plików pickle dla backendu

import pickle

try:
    logging.info("Rozpoczynanie eksportu modeli LightGBM oraz Ensemble do plików pickle...")

    # eksport modelu LightGBM
    sciezka_lgb = "static/model_lightgbm.pkl"
    with open(sciezka_lgb, "wb") as plik_lgb:
        pickle.dump(modele_wytrenowane['LightGBM'], plik_lgb)
    logging.info(f"Pomyślnie zapisano model LightGBM w: {sciezka_lgb}")

    # eksport modelu Ensemble
    sciezka_ensemble = "static/model_ensemble.pkl"
    with open(sciezka_ensemble, "wb") as plik_ens:
        pickle.dump(modele_wytrenowane['Ensemble'], plik_ens)
    logging.info(f"Pomyślnie zapisano model Ensemble w: {sciezka_ensemble}")

    print("\npliki pickle gotowe")

except Exception as error_pickle:
    logging.error(f"Błąd podczas zapisu plików pickle: {error_pickle}")