import json
import pickle
import numpy as np

model_name = 'flower-v1.pkl'
model_pk = pickle.load(open(model_name, 'rb'))

def predict(event= None, context= None):
	
	body = {
        "message": "OK",
    }
	
	if 'queryStringParameters' in event.keys():
        
		params = event['queryStringParameters']
		
		sepal_length = float(params['sepal_length'])
		sepal_width = float(params["sepal_width"])
		petal_length = float(params["petal_length"])
		petal_width = float(params["petal_width"])
		
		data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
		
		prediction = model_pk.predict(data)
		
		body['prediction'] = prediction
		
	else:
	
		body['message'] = 'queryStringParameters not in event.'

	response = {
        "statusCode": 200,
        "body": prediction[0],
        "headers": {
            "Content-Type": 'application/json',
            "Access-Control-Allow-Origin": "*"
        }
    }
	
	return response

data = {"queryStringParameters": {"sepal_length": 200000, "sepal_width": 10, "petal_length": 4, "petal_width": 1}}
print(predict(data))