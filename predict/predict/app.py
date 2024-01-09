from flask import Flask, request, render_template
import joblib
from predict.predict import run as run_predict

artefacts_path = "C:/Users/antoi/OneDrive/Documents/COURS_2023_S2/POC_Prod/poc-to-prod-capstone/train/data/artefacts/2024-01-09-12-47-54"
model = run_predict.TextPredictionModel.from_artefacts(artefacts_path)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    user_text = ''
    prediction = ''
    if request.method == 'POST' and 'user_text' in request.form:
        user_text = request.form['user_text']
        predictions = model.predict([user_text])
        index_to_labels = {v: k for k, v in model.labels_to_index.items()}
        prediction = [index_to_labels.get(idx, 'Unknown label') for idx in predictions[0]]
        print(f'Prediction: {prediction}')  # Print the prediction
    return render_template('index.html', prediction=prediction, user_text=user_text)

if __name__ == '__main__':
    app.run(debug=True)