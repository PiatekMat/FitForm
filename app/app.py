from flask import Flask, render_template, request
from predictor import WeightPredictor
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Załadowanie modelu tylko raz przy uruchomieniu aplikacji
predictor = WeightPredictor("models/model_lightgbm.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Pobranie danych z formularza

        weight = float(request.form["weight"])
        kcal = float(request.form["kcal"])
        protein = float(request.form["protein"])
        burned = float(request.form["burned"])
        cardio = float(request.form["cardio"])
        strength = int(request.form["strength"])
        steps = int(request.form["steps"])
        days = int(request.form["days"])

        # Wykonanie predykcji

        weights = predictor.symulacja_wagi(
            start_weight=weight,
            kcal=kcal,
            bialko=protein,
            spalone_kcal=burned,
            cardio=cardio,
            silowy=strength,
            kroki=steps,
            dni=days
        )

        # Wygenerowanie wykresu

        predictor.save_plot(weights)

        # Wyświetlenie wyniku

        return render_template(
            "results.html",
            prediction=round(weights[-1], 2),
            weights=weights,
            days=days,
            plot="prediction.png"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)