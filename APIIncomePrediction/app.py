from flask import Flask, request, jsonify
import pandas as pd
import pickle


app = Flask(__name__)

# Loading transformer
with open(r'Projects\APIIncomePrediction\transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

# Loading model
with open(r'Projects\APIIncomePrediction\model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Getting input
    input_json = request.get_json()
    new_X = pd.DataFrame([input_json])

    # Encoding and Scaling
    new_X = transformer.transform(new_X)

    # Prediction
    new_y = model.predict(new_X)[0]
    confidence = model.predict_proba(new_X)[0][1]

    if new_y == 0:
        prediction = '<=50K'
    else:
        prediction = '>50K'

    return jsonify({"prediction": prediction,
                    "confidence": confidence})


if __name__ == '__main__':
    app.run(debug=True)

