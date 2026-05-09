from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        avg_kcal = request.form.get("avg_kcal")
        avg_burned_kcal = request.form.get("avg_burned_kcal")
        weight = request.form.get("weight")
        gender = request.form.get("gender")

        return f"""
        kcal: {avg_kcal}<br>
        burned: {avg_burned_kcal}<br>
        weight: {weight}<br>
        gender: {gender}
        """

    return render_template("index.html")

#komentarz

if __name__ == "__main__":
    app.run(debug=True)