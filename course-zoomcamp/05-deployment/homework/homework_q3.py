
import pickle

# Path to Model and Dict Vectorizer
dict_vect_path = "./dv.bin"
model_path = "./model1.bin"

# Customer
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

# Load the model 
def load_file(path_to_file):
    with open(path_to_file, 'rb') as f_in:
        obj = pickle.load(f_in)
    return obj

# Predict method
def predict(customer, model, dv):
    x = dv.transform([customer])
    pred = model.predict_proba(x)[0,1]

    churn = pred >= 0.5

    result = {
        'churn probability':float(pred),
        'churn': bool(churn)
    }

    return result

def run():
    model = load_file(model_path)
    dv = load_file(dict_vect_path)

    pred = predict(customer, model=model,dv=dv)
    print(pred)

if __name__ == "__main__":
    run()
