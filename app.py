from flask import Flask, request, jsonify
from helpers.preprocessing import PreProcessing
import joblib

vectorizer = joblib.load("modeling/vectorizer.save")
label_encoder = joblib.load("modeling/label_encoder.save")
model = joblib.load("modeling/decision_tree_model.pkl")
preprocessing = PreProcessing()


# create an instance of Flask
app = Flask(__name__, template_folder="public")


@app.route("/")
def home():
    return "Bipolar Prediction"


@app.route("/api/predict/bipolar", methods=["POST"])
def prediction():
    try:
        symptoms = request.json["symptoms"]
        symptoms_preprocess = preprocessing.transform(symptoms)

        X = vectorizer.transform([symptoms_preprocess]).toarray()
        prediction = model.predict(X)
        prediction = label_encoder.inverse_transform(prediction)[0]
        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "symptoms": symptoms,
                        "word_used_to_predict": symptoms_preprocess,
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
