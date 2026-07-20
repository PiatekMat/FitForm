from flask import Flask, render_template, request
from predictor import WeightPredictor
import os
from training.generator import generate_training_plan

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
        gender = request.form["plec"]

        # Wykonanie predykcji

        weights = predictor.symulacja_wagi(
        days,
        weight,
        kcal,
        protein,
        burned,
        cardio,
        strength,
        steps
        )   
        predicted_change = weights[-1] - weights[0]
        training_plan = generate_training_plan(
        gender=gender,
        predicted_weight_change=predicted_change,
        training_days=strength
)

        return render_template(
            "results.html",
            weights=weights,
            training_plan=training_plan,
            start_weight=weight,
            end_weight=weights[-1]
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)