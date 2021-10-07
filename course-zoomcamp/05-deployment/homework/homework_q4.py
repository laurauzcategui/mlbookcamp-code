
import pickle
from flask import Flask
from flask import request, jsonify

app = Flask("churn")

# Path to Model and Dict Vectorizer
dict_vect_path = "./dv.bin"
model_path = "./model2.bin"

# Load the model 
def load_file(path_to_file):
    with open(path_to_file, 'rb') as f_in:
        obj = pickle.load(f_in)
    return obj

model = load_file(model_path)
dv = load_file(dict_vect_path)

@app.route('/predict', methods=["POST"])
def predict():
    customer = request.get_json()
    x = dv.transform([customer])
    pred = model.predict_proba(x)[0,1]

    churn = pred >= 0.5

    result = {
        'churn probability':float(pred),
        'churn': bool(churn)
    }
    
    return jsonify(result) 

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

