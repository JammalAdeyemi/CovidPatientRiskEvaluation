import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("my_classification_model_2")

df = pd.read_csv('./Data/Covid-19.csv')

# Splitting the datasets and choosing 10 columns that are correlated with the DEATH Column
X = SelectKBest(k=14, score_func=f_classif).fit_transform(df.drop(['DEATH'],axis=1), df['DEATH'])
y = df['DEATH']

scaler = RobustScaler()
X = scaler.fit_transform(X)

# Splitting the data into train and test, since the data is imbalance I would Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

with mlflow.start_run(run_name='My model experiment 3') as run:

    gb = GradientBoostingClassifier()
    gb.fit(X_train,y_train)
    mlflow.sklearn.log_model(gb, "GradientBoostingClassifier-model")

    # log model performance
    accuracy = gb.score(X_test,y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Gradient Boosting Classifier Accuracy: {accuracy}")

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    mlflow.end_run()
    print(f'artifact_uri = {mlflow.get_artifact_uri()}')
    print(f'runID: {run_id}')