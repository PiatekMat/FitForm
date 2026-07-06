import pickle
import pandas as pd
import matplotlib.pyplot as plt


class WeightPredictor:

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.columns = [
            "zjedzone_kcal",
            "bialko_g",
            "spalone_kcal",
            "cardio_min",
            "trening_silowy",
            "kroki",
            "waga_czczo",
            "dzien"
        ]

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


    def save_plot(self, weights, path="static/prediction.png"):

        plt.figure(figsize=(10,5))

        plt.plot(
            range(len(weights)),
            weights,
            marker="o"
        )

        plt.title("Prognoza masy ciała")
        plt.xlabel("Dzień")
        plt.ylabel("Waga [kg]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return path