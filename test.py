import joblib
new_data=[]

model = joblib.load('xgb_customer_acquisition_pipeline.joblib')
predict=model.predict(new_data)
production_prediction = predict[0]
probability = model.predict_proba(new_data)[0][1]
accuracy = model.score(new_data, production_prediction)
gini = 2 * accuracy - 1
precision = model.score(new_data, production_prediction)