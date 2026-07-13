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
            "bilans_kcal",
            "bilans_pct",
            "kcal_na_kg",
            "bialko_na_kg",
            "spalone_z_silowym",
            "cardio_i_silowy",
            "aktywnosc_total",
            "dzien_tygodnia",
            "weekend",
            "zjedzone_kcal_ma3",
            "zjedzone_kcal_ma7",
            "spalone_kcal_ma3",
            "spalone_kcal_ma7",
            "bilans_kcal_ma3",
            "bilans_kcal_ma7"]
        
    def symulacja_wagi(self, dni, start_weight, kcal, bialko, spalone_kcal, cardio, silowy, kroki):
        lista_wag = [start_weight]
        obecna_waga = start_weight

        for d in range(1, dni + 1):

            bilans_kcal = kcal - spalone_kcal

            dane_wiersza = {
                "zjedzone_kcal": kcal,
                "bialko_g": bialko,
                "spalone_kcal": spalone_kcal,
                "cardio_min": cardio,
                "trening_silowy": silowy,
                "kroki": kroki,
                "bilans_kcal": bilans_kcal,
                "bilans_pct": bilans_kcal / spalone_kcal if spalone_kcal > 0 else 0,
                "kcal_na_kg": kcal / obecna_waga if obecna_waga > 0 else 0,
                "bialko_na_kg": bialko / obecna_waga if obecna_waga > 0 else 0,
                "spalone_z_silowym": spalone_kcal * silowy,
                "cardio_i_silowy": cardio * silowy,
                "aktywnosc_total": cardio + kroki / 100,
                "dzien_tygodnia": d % 7,
                "weekend": 1 if (d % 7) >= 5 else 0,
                "zjedzone_kcal_ma3": kcal,
                "zjedzone_kcal_ma7": kcal,
                "spalone_kcal_ma3": spalone_kcal,
                "spalone_kcal_ma7": spalone_kcal,
                "bilans_kcal_ma3": bilans_kcal,
                "bilans_kcal_ma7": bilans_kcal,
                }

            wejscie = pd.DataFrame([dane_wiersza])[self.columns].astype("float32")

            roznica = self.model.predict(wejscie)[0]
            obecna_waga += roznica
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