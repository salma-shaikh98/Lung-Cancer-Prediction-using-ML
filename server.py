from flask import Flask, render_template, request, jsonify,send_file
import pandas as pd
import joblib  

app = Flask(__name__, static_folder='static')

# Loading the trained models
model_lr = joblib.load('logistic_regression_model.pkl')
model_svm = joblib.load('svm_model.pkl')
model_knn = joblib.load('knn_model.pkl')
model_dt = joblib.load('decision_tree_model.pkl')

# Defining feature names for each model
feature_names_lr =  ['air_pollution', 'alcohol_consumption', 'dust_allergy', 'occ_hazard','genetic_risk','chronic_lung_disease','balanced_diet','obesity','smoking','passive_smoker','chest_pain','coughing_blood','fatigue','weight_loss','short_breath','wheezing','swollow_difficulty','clubbed_finger','frequent_cold','dry_cough','snoring']  # Features for Logistic Regression model

feature_names_svm =  ['air_pollution', 'alcohol_consumption', 'dust_allergy', 'occ_hazard','genetic_risk','chronic_lung_disease','balanced_diet','obesity','smoking','passive_smoker','chest_pain','coughing_blood','fatigue','weight_loss','short_breath','wheezing','swollow_difficulty','clubbed_finger','frequent_cold','dry_cough','snoring']  # Features for SVM model

feature_names_knn =  ['air_pollution', 'alcohol_consumption', 'dust_allergy', 'occ_hazard','genetic_risk','chronic_lung_disease','balanced_diet','obesity','smoking','passive_smoker','chest_pain','coughing_blood','fatigue','weight_loss','short_breath','wheezing','swollow_difficulty','clubbed_finger','frequent_cold','dry_cough','snoring']  # Features for KNN model

feature_names_dt =  ['air_pollution', 'alcohol_consumption', 'dust_allergy', 'occ_hazard','genetic_risk','chronic_lung_disease','balanced_diet','obesity','smoking','passive_smoker','chest_pain','coughing_blood','fatigue','weight_loss','short_breath','wheezing','swollow_difficulty','clubbed_finger','frequent_cold','dry_cough','snoring']   # Features for Decision Tree model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dataset')
def display_dataset():
    # Read Excel file into a DataFrame
    df = pd.read_excel('cancer_data.xlsx')  # Replace 'path_to_your_excel_file.xlsx' with the actual path to your Excel file

    # Convert DataFrame to list of dictionaries for easier processing in the template
    data = df.to_dict(orient='records')

    return render_template('dataset.html', data=data)

@app.route('/documentation')
def display_documentation():
    # Replace 'path_to_your_pdf_file.pdf' with the actual path to your PDF file
    return send_file('synopsis.pdf', as_attachment=False)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("Form Data:", request.form)
        
        # Extracting form data
        air_pollution = float(request.form['air_pollution'])
        alcohol_consumption = float(request.form['alcohol_consumption'])
        dust_allergy = float(request.form['dust_allergy'])
        occ_hazard = float(request.form['occ_hazard'])
        genetic_risk = float(request.form['genetic_risk'])
        chronic_lung_disease = float(request.form['chronic_lung_disease'])
        balanced_diet = float(request.form['balanced_diet'])
        obesity = float(request.form['obesity'])
        smoking = float(request.form['smoking'])
        passive_smoker = float(request.form['passive_smoker'])
        chest_pain = float(request.form['chest_pain'])
        coughing_blood = float(request.form['coughing_blood'])
        fatigue = float(request.form['fatigue'])
        weight_loss = float(request.form['weight_loss'])
        short_breath = float(request.form['short_breath'])
        wheezing = float(request.form['wheezing'])
        swollow_difficulty = float(request.form['swollow_difficulty'])
        clubbed_finger = float(request.form['clubbed_finger'])
        frequent_cold = float(request.form['frequent_cold'])
        dry_cough = float(request.form['dry_cough'])
        snoring = float(request.form['snoring'])

        # Preparing the data for each model
        data = pd.DataFrame({
            'air_pollution': [air_pollution],
            'alcohol_consumption': [alcohol_consumption],
            'dust_allergy': [dust_allergy],
            'occ_hazard': [occ_hazard],
            'genetic_risk': [genetic_risk],
            'chronic_lung_disease': [chronic_lung_disease],
            'balanced_diet': [balanced_diet],
            'obesity': [obesity],
            'smoking': [smoking],
            'passive_smoker': [passive_smoker],
            'chest_pain': [chest_pain],
            'coughing_blood': [coughing_blood],
            'fatigue': [fatigue],
            'weight_loss': [weight_loss],
            'short_breath': [short_breath],
            'wheezing': [wheezing],
            'swollow_difficulty': [swollow_difficulty],
            'clubbed_finger': [clubbed_finger],
            'frequent_cold': [frequent_cold],
            'dry_cough': [dry_cough],
            'snoring': [snoring],
        })

        # Extract selected model
        selected_model = request.form['model']

        # Making predictions using the selected model
        if selected_model == 'logistic_regression':
            prediction = model_lr.predict(data[feature_names_lr])
        elif selected_model == 'svm':
            prediction = model_svm.predict(data[feature_names_svm])
        elif selected_model == 'knn':
            prediction = model_knn.predict(data[feature_names_knn])
        elif selected_model == 'decision_tree':
            prediction = model_dt.predict(data[feature_names_dt])
        else:
            return jsonify({'error': 'Invalid model selection'})

        # Returning the predictions
        return jsonify({'prediction': prediction.tolist()})
    else:
        # Render the form for prediction
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)