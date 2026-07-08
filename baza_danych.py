import random
import pandas as pd
from faker import Faker

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

fake = Faker('pl_PL')

# Profile logiczne, bez wrzucania do końcowej tabeli, wraz z dwoma dodatkowymi, częstymi przypadkami w dietetyce i treningu
PROFILE = {
    'redukcja_zdrowy': {'kroki_min': 8000, 'kroki_max': 15000, 'kcal_mod': 0.80, 'bmr_mod': 1.0, 'szansa_silowy': 0.7},
    'redukcja_hashimoto': {'kroki_min': 4000, 'kroki_max': 8000, 'kcal_mod': 0.85, 'bmr_mod': 0.85,
                           'szansa_silowy': 0.4},
    'masa_zdrowy': {'kroki_min': 6000, 'kroki_max': 10000, 'kcal_mod': 1.15, 'bmr_mod': 1.0, 'szansa_silowy': 0.8},
    'kanapowiec_zdrowy': {'kroki_min': 2000, 'kroki_max': 5000, 'kcal_mod': 1.00, 'bmr_mod': 1.0, 'szansa_silowy': 0.1},
    'kanapowiec_io': {'kroki_min': 1500, 'kroki_max': 4000, 'kcal_mod': 1.05, 'bmr_mod': 0.90, 'szansa_silowy': 0.1}
}


def generuj_baze(liczba_osob=100, dni_na_osobe=77):
    all_logs = []

    for i in range(liczba_osob):
        plec = random.choice(['M', 'K'])
        typ_profilu = random.choice(list(PROFILE.keys()))
        cfg = PROFILE[typ_profilu]

        waga = random.uniform(55, 110)
        wzrost = random.randint(160, 195) if plec == 'M' else random.randint(150, 175)
        wiek = random.randint(20, 50)

        # Generator cofania czasu
        start_date = pd.Timestamp.now() - pd.Timedelta(days=dni_na_osobe)

        for d in range(dni_na_osobe):
            data_wpisu = start_date + pd.Timedelta(days=d)

            # Wzór BMR
            if plec == 'M':
                bmr = (10 * waga) + (6.25 * wzrost) - (5 * wiek) + 5
            else:
                bmr = (10 * waga) + (6.25 * wzrost) - (5 * wiek) - 161
            bmr = bmr * cfg['bmr_mod']

            kroki = random.randint(cfg['kroki_min'], cfg['kroki_max'])
            cardio = random.choice([0, 0, 20, 30, 41, 45]) if cfg['kroki_min'] > 4000 else 0

            # Zamiana treningu na smallint (0 lub 1)
            silowy = 1 if random.random() < cfg['szansa_silowy'] else 0

            spalone_kcal = int(bmr + (kroki * 0.04) + (cardio * 7) + (silowy * 250))
            zjedzone_kcal = int(spalone_kcal * cfg['kcal_mod'])

            bialko = round(waga * random.uniform(1.6, 2.2), 2) if silowy == 1 else round(
                waga * random.uniform(1.0, 1.4), 2)

            bilans = zjedzone_kcal - spalone_kcal
            delta_tluszcz = bilans / 7700

            all_logs.append({
                'data_wpisu': data_wpisu.strftime('%Y-%m-%d'),
                'zjedzone_kcal': int(zjedzone_kcal),
                'bialko_g': float(bialko),
                'spalone_kcal': int(spalone_kcal),
                'cardio_min': int(cardio),
                'trening_silowy': int(silowy),
                'kroki': int(kroki),
                'waga_czczo': float(round(waga, 2))
            })

            waga += delta_tluszcz

    df_fake = pd.DataFrame(all_logs)

    kolejnosc_kolumn = [
        'data_wpisu', 'zjedzone_kcal', 'bialko_g',
        'spalone_kcal', 'cardio_min', 'trening_silowy', 'kroki', 'waga_czczo'
    ]
    return df_fake[kolejnosc_kolumn]


df = generuj_baze(liczba_osob=100, dni_na_osobe=77)

print("\n PODGLĄD PRZYKŁADOWEJ OSOBY" )
print(df.head(10))
