from flask import Flask, render_template, request
from predictor import WeightPredictor

app = Flask(__name__)

# Załadowanie modelu tylko raz przy uruchomieniu aplikacji
predictor = WeightPredictor("models/model_ensemble.pkl")


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

        weights = predictor.predict_weight(
            start_weight=weight,
            kcal=kcal,
            protein=protein,
            burned=burned,
            cardio=cardio,
            strength=strength,
            steps=steps,
            days=days
        )

        # Wygenerowanie wykresu

        predictor.save_plot(weights)

        # Wyświetlenie wyniku

        return render_template(
            "result.html",
            prediction=round(weights[-1], 2),
            weights=weights,
            days=days,
            plot="prediction.png"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)