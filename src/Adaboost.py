import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, r2_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Project")

df = pd.read_csv('./Data/Covid-19.csv')

with mlflow.start_run(run_name="AdaBoost Classifier") as run:

    # Splitting the datasets and choosing 14 columns that are correlated with the DEATH Column
    y = df['DEATH']
    X = SelectKBest(k=14, score_func=f_classif).fit_transform(df.drop(['DEATH'],axis=1), df['DEATH'])

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

    ada_boost = AdaBoostClassifier(learning_rate=0.5748335028625987, n_estimators=120,random_state=42)
    ada_boost.fit(X_train, y_train)
    y_preds = ada_boost.predict(X_test)

    # Log model artifacts (e.g. model files)
    mlflow.sklearn.log_model(ada_boost, "model")

    accuracy = ada_boost.score(X_test,y_test)
    r2 = r2_score(y_test,y_preds)

    print(f"AdaBoost Accuracy: {accuracy}")
    print(f"AdaBoost Classifier R2 Score: {r2}")
    print(classification_report(y_test, y_preds))

    # Log model metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("r2_score", r2)
    # mlflow.log_metric("classification_report", classification_report(y_test, y_preds))

    # Log a classification report as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_preds))
    mlflow.log_artifact("classification_report.txt")

    # Log a confusion matrix as an artifact
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, fmt=".0f")
    plt.title("AdaBoost Classifier Confusion Matrix",fontsize=18, color="b")
    plt.savefig("adaboost_conf_matrix.jpeg")
    mlflow.log_artifact("adaboost_class_conf_matrix.jpeg")

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id
    mlflow.end_run()
    print(f'artifact_uri = {mlflow.get_artifact_uri()}')
    print(f'runID: {run_id}')