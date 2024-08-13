from app import app
from flask import render_template, request


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        columns = [
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "parking",
            "prefarea",
            "furnishingstatus",
        ]

        checkboxes = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]

        X_input = []
        for column in columns:
            if column not in checkboxes:
                val = request.form.get(column, None)
                if column == "furnishingstatus":
                    X_input.append(val)
                else:
                    X_input.append(float(val))
            else:
                X_input.append(1 if column in request.form.getlist("feature") else 0)

        print(X_input)

    return render_template("predict.html")
