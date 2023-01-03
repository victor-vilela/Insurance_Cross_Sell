import pickle
import pandas as pd
from flask import Flask, request, Response
from health_insurance import HealthInsurance
import os

# loading model
model = pickle.load(open( 'models/model_linear_regression.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    
    if test_json: # data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys() )
    
        # instanciate health insurance class
        pipeline = HealthInsurance()

        df1 = pipeline.data_engineering(test_raw)

        df2 = pipeline.data_preparation(df1)

        df_response = pipeline.get_prediction(model, test_raw, df2)

        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)

