from flask import Flask, request, jsonify, render_template
from helpers.preprocessing import PreProcessing
from flask_cors import cross_origin

import joblib

vectorizer = joblib.load("modeling/vectorizer.save")
label_encoder = joblib.load("modeling/label_encoder.save")
model = joblib.load("modeling/decision_tree_model.pkl")
preprocessing = PreProcessing()


# create an instance of Flask
app = Flask(__name__, template_folder="public")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict/bipolar", methods=["POST"])
@cross_origin()
def prediction():
    try:
        symptoms = request.json["symptoms"]
        symptoms_preprocess = preprocessing.transform(symptoms)

        X = vectorizer.transform(symptoms_preprocess)
        prediction = model.predict(X)
        prediction = label_encoder.inverse_transform(prediction)[0]
        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "symptoms": symptoms,
                        "word_used_to_predict": symptoms_preprocess[0],
                        "prediction": prediction,
                    },
                }
            ),
            201,
        )
    except:
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": 500,
                        "messsage": "Something wrong, please try again!",
                    },
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
