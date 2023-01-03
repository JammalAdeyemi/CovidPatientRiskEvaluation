import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, r2_score, classification_report

mlflow.set_experiment("Project")

df = pd.read_csv('./Data/Covid-19.csv')

with mlflow.start_run(run_name='Logistic Regression(GridSearchCV)') as run:

    # Splitting the datasets and choosing 14 columns that are correlated with the DEATH Column
    X = SelectKBest(k=14, score_func=f_classif).fit_transform(df.drop(['DEATH'],axis=1), df['DEATH'])
    y = df['DEATH']

    # Log a parameter
    mlflow.log_param("feature_selection", "SelectKBest")

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # Splitting the data into train and test, since the data is imbalance I would Undersample the majority class
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Log model parameters
    mlflow.log_param("random_state", 42)
    mlflow.log_param("scaler", "RobustScaler")
    mlflow.log_param("resampler", "RandomUnderSampler")

    logreg_cv = LogisticRegression(C= 0.001, penalty = 'l2', random_state=42)
    logreg_cv.fit(X_train,y_train)
    y_preds = logreg_cv.predict(X_test)

    # Log model artifacts (e.g. model files)
    mlflow.sklearn.log_model(logreg_cv, "Logistic-regression-gridsearchcv-model")

    # log model performance
    accuracy = logreg_cv.score(X_test,y_test)
    r2 = r2_score(y_test,y_preds)
    print(f"GridsearchCV Logistic Regression Accuracy: {accuracy}")
    print(f"Logistic Regression Classifier r2_Score: {r2}")
    print(classification_report(y_test, y_preds))

    # Log model metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("r2_score", r2)

    # Log a classification report as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_preds))
    mlflow.log_artifact("classification_report.txt")

    # Log a confusion matrix as an artifact
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, fmt=".0f")
    plt.title("Logistic Regression Confusion Matrix",fontsize=18, color="b")
    plt.savefig("Logreg_gridsearchcv_conf_matrix.jpeg")
    mlflow.log_artifact("Logreg_gridsearchcv_conf_matrix.jpeg")

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    mlflow.end_run()
    print(f'artifact_uri = {mlflow.get_artifact_uri()}')
    print(f'runID: {run_id}')