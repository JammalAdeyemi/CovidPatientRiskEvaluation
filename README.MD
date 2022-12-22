# Covid Risk Predictor

The goal of this project is to build a machine learning model that can predict the risk level of a Covid-19 patient based on their current symptoms, status, and medical history. This model will be trained on a dataset of Covid-19 patient records and will be able to classify patients as either low risk or high risk.

## Getting Started

To set up this project on your local machine, you will need to install the following dependencies:
- Python 3.6 or higher
- NumPy
- pandas
- scikit-learn

You can install these dependencies using pip:
* pip install numpy pandas scikit-learn

## Usage

To use the model, you will need to pass in the patient's current symptoms, status, and medical history as input features. The model will output a prediction of the patient's risk level as either low or high.

Here is an example of how to use the model:
* from risk_predictor import predict_risk

### Define the input features for a patient
patient_features = {

"symptoms": ["cough", "fever"],
"status": "hospitalized",
"medical_history": ["diabetes", "heart disease"]

}

### Predict the patient's risk level
risk_prediction = predict_risk(patient_features)

print(risk_prediction) # Output: "high"
