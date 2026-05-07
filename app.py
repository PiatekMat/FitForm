from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        avg_kcal = request.form.get("avg_kcal")
        avg_burned_kcal = request.form.get("avg_burned_kcal")
        weight = request.form.get("weight")
        gender = request.form.get("gender")
        training_days = request.form.get("training_days")

        return f"""
        kcal: {avg_kcal}<br>
        spalone: {avg_burned_kcal}<br>
        waga: {weight}<br>
        płeć: {gender}<br>
        dni treningowe: {training_days}
        """

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)