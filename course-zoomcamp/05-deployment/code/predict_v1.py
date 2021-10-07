
import pickle
from flask import Flask
from flask import request, jsonify

app = Flask("churn")

C = 1.0
output_file = f'model_C={C}.bin'

# Load the model 
with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

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

