import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.ensemble import RandomForestRegressor
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

# ---------CONFIG LOGGER----------
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

# -------- MODELOWANIE RÓŻNICOWE------

df = df.sort_values(by=['user_id', 'data_wpisu'])

# utworzenie wagi z następnego dnia
df['waga_jutro'] = df.groupby('user_id')['waga_czczo'].shift(-1)

# Y staje się różnica wagi między jutrem a dzisiaj
df['roznica_wagi'] = df['waga_jutro'] - df['waga_czczo']

# usuniecie wierszy, dla których nie jest znana różnica (ostatni dzień każdego użytkownika)
df_clean = df.dropna(subset=['roznica_wagi']).copy()

# niewykorzystano wagi na czczo, wczesniej negatywnie wplynela na predykcje
X = df_clean.drop(['data_wpisu', 'waga_czczo', 'waga_jutro', 'roznica_wagi', 'id', 'user_id'], axis=1, errors='ignore')
y = df_clean['roznica_wagi']

kolumny_X = X.columns.tolist()

user_groups = df_clean['user_id']
group_cv = GroupKFold(
    n_splits=4)  # GroupKFold grwanatuje że każda grupa trafia w całości albo do trenowania, albo do testowania

# ----------XGBOOST--------------
model_xgb = xgb.XGBRegressor(
    n_estimators=50,  # mało drzew, bo mała ilość danych
    max_depth=3,  # płytsze drzewo - zapobiega overfittingowi
    learning_rate=0.1,
    objective='reg:squarederror',
    reg_alpha=0.1,  # regularyzacja L1 lasso
    reg_lambda=1  # regularyzacja L2 ridge
)


# ----------- LIGHTGBM ----------------
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

# -----CATBOOST--------
model_cat = CatBoostRegressor(
    iterations=200,
    depth=5,
    learning_rate=0.1,
    # cat_features= cat_features_indices, cat_features_indices=[0,3] jesli mamy dane kategoryczne podajmy ich kolumny
    loss_function='RMSE',
    verbose=0
)

# -----Random Forest-----
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
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
    ('model_svm', SVR(kernel='rbf', C=10.0, epsilon=0.1))  # rbf pozwala uczyc sie krzywych
    # C siła kary - im wyzsze tym model stara się bardziej dopasowac do każdego punktu
    # epsilon - szerokość "tuby" błędu, którą model ignoruje
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
logging.info("Modele zostały pomyślnie wytrenowane.")


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

# ----------------WYKRES PORÓWNAWCZY-----------------

try:
    logging.info("Generowanie wykresu dopasowania dla wszystkich 10 modeli...")

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
            'Decision Tree': [waga_start_user]
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
            'Decision Tree': model_dt.predict(X_user)
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

        plt.title(
            f"Analiza porównawcza modeli",
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
        ('SVM (RBF)', pipeline_svm), ('KNN', pipeline_knn), ('Decision Tree', model_dt)
    ]

    # pobrano dane dla najczęstszego użytkownika (zbiór testowy)
    najczestszy_user = df_clean['user_id'].value_counts().index[0]
    df_user = df_clean[df_clean['user_id'] == najczestszy_user].sort_values(by='data_wpisu').copy()
    X_test_user = df_user[kolumny_X]
    y_test_user = df_user['roznica_wagi']

    wiersze = []

    """
    Zastosowano walidację krzyżową GroupKFold (odmianę K-Fold Cross-Validation).
    Klasyczny podział na dwie części byłby zbyt losowy i ryzykowny dla małego zbioru danych. 

    Algorytm dzieli bazę na 4 części (foldy) według 'user_id'. Cała historia jednego użytkownika 
    trafia w całości albo do treningu, albo do testu. Model jest więc sprawdzany na osobach, 
    których wcześniej "nie widział". Średnia z 4 prób daje obiektywny błąd RMSE.
    """

    for nazwa, model in lista_modeli:
        cv = cross_validate(model, X, y, cv=group_cv, groups=user_groups, scoring=scoring_metrics, n_jobs=1)

        rmse = -cv['test_rmse']
        mae = -cv['test_mae']
        r2 = cv['test_r2']

        # obliczenia na wydzielonym zbiorze testowym dla konkretnego użytkownika
        predykcje_testowe = model.predict(X_test_user)
        test_rmse = np.sqrt(np.mean((y_test_user - predykcje_testowe) ** 2))
        test_mae = np.mean(np.abs(y_test_user - predykcje_testowe))

        # obliczanie R2 dla testu
        wariancja_calkowita = np.sum((y_test_user - np.mean(y_test_user)) ** 2)
        if wariancja_calkowita > 0:
            test_r2 = 1 - (np.sum((y_test_user - predykcje_testowe) ** 2) / wariancja_calkowita)
        else:
            test_r2 = 0.0

        wiersze.append({
            'Model': nazwa,
            'RMSE mean CV [kg]': round(np.mean(rmse), 4),
            'RMSE std CV [kg]': round(np.std(rmse), 4),
            'RMSE pct of range CV [kg]': round(np.max(rmse) - np.min(rmse), 4),
            'MAE mean CV [kg]': round(np.mean(mae), 4),
            'MAE std CV [kg]': round(np.std(mae), 4),
            'R2 mean CV [kg]': round(np.mean(r2), 4),
            'R2 std CV [kg]': round(np.std(r2), 4),
            'Test RMSE [kg]': round(test_rmse, 4),
            'Test MAE [kg]': round(test_mae, 4),
            'Test R2 [kg]': round(test_r2, 4)
        })

    df = pd.DataFrame(wiersze)

    # sortowanie tabeli
    df = df.sort_values(by='MAE mean CV [kg]', ascending=True).reset_index(drop=True)

    logging.info(
        "\n" + "=" * 110 + "\n" + "Tabela porównawcza" + "\n" + "=" * 110)
    logging.info(f"\n{df.to_string(index=False)}")

    sciezka_csv = "static/tabela_porownawcza.csv"
    df.to_csv(sciezka_csv, index=False, encoding='utf-8-sig')
    logging.info(f"Zapisano tabelę w: {sciezka_csv}")

except Exception as e:
    logging.error(f"Błąd podczas generowania tabeli: {e}")

# ----------- PFI ORAZ SHAP -------------
from sklearn.inspection import permutation_importance
import shap

try:
    logging.info("Rozpoczęcie analizy PFI i SHAP dla modelu LightGBM...")

    # ----- PFI --------
    logging.info("Obliczanie PFI...")
    # obliczanie PFI na pełnym zbiorze danych X, y dla modelu lightgbm
    pfi_wynik = permutation_importance(model_lgb, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    # sortowanie cech
    indeksy_posortowanej_waznosci = pfi_wynik.importances_mean.argsort()
    cechy_sortowane = [kolumny_X[i] for i in indeksy_posortowanej_waznosci]
    waznosc_sortowana = pfi_wynik.importances_mean[indeksy_posortowanej_waznosci]

    # generowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.barh(cechy_sortowane, waznosc_sortowana, color='#008080', edgecolor='black', alpha=0.8)
    plt.xlabel("Spadek dokładności modelu (Wzrost błędu po przetasowaniu cechy)", fontsize=11)
    plt.title("Globalna ważność cech: Permutation Feature Importance (PFI)\nModel: LightGBM", fontsize=13,
              fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6, axis='x')

    sciezka_pfi = "static/pfi_lightgbm.png"
    plt.tight_layout()
    plt.savefig(sciezka_pfi, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres PFI w: {sciezka_pfi}")

    # ---------- SHAP ---------
    logging.info("Obliczanie wartości SHAP...")
    # explainer dedykowany dla modeli opartych na drzewach
    explainer = shap.TreeExplainer(model_lgb)
    shap_values= explainer(X)

    # wykres globalny SHAP (Summary Plot) - rozkład i kierunek wpływu cech
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Wpływ zmiennych na dobową zmianę wagi", fontsize=13, fontweight='bold',
              pad=25)

    sciezka_shap_summary = "static/shap_summary.png"
    plt.tight_layout()
    plt.savefig(sciezka_shap_summary, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano globalny wykres SHAP Summary w: {sciezka_shap_summary}")

    # wykres lokalny SHAP

    # wybieranie pierwszego dnia dla najczęstszego użytkownika
    idx_konkretnego_dnia = df_user.index[0]
    pos_in_X = X.index.get_loc(idx_konkretnego_dnia)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[pos_in_X], show=False)
    plt.title(
        f"Lokalna interpretacja pojedynczej predykcji",
        fontsize=12, fontweight='bold', pad=25)

    sciezka_shap_waterfall = "static/shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(sciezka_shap_waterfall, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano lokalny wykres SHAP Waterfall w: {sciezka_shap_waterfall}")

    logging.info("Wszystkie analizy XAI zostały pomyślnie wygenerowane i zapisane w folderze static")

except ModuleNotFoundError as e:
    logging.error(f"Brak biblioteki: {e}")
except Exception as e:
    logging.error(f"Błąd podczas generowania analizy: {e}")
