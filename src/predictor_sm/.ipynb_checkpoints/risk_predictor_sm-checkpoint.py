"""Sagemaker Covid Risk Predictor."""
import os
import logging
import joblib
import numpy as np
from io import BytesIO
from sklearn.ensemble import VotingClassifier

logger = logging.getLogger(__name__)

MODEL_FILE_NAME = "voting_classifier.pkl"


def model_fn(model_dir):
    """Deserialize and return fitted model."""
    logger.info("Loading model...")
    file_name = os.path.join(model_dir, MODEL_FILE_NAME)
    loaded_model: VotingClassifier = joblib.load(open(file_name, "rb"))
    logger.info("Model loaded successfully.")
    return loaded_model


def input_fn(request_body, request_content_type):
    """The SageMaker model server receives the request data body and the content type,
    and invokes the `input_fn`.

    Return a NumPy (an object that can be passed to predict_fn).
    """
    if request_content_type == "application/x-npy":
        array = np.load(BytesIO(request_body), allow_pickle=True)
        return array
    else:
        raise ValueError("Content type {} is not supported.".format(request_content_type))


def predict_fn(input_data, model):
    """SageMaker model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array where the first columns are predictions
    and the second columns are probabilities."""
    prediction = model.predict(input_data)
    pred_prob = model.predict_proba(input_data)
    return np.array([prediction, pred_prob])


def output_fn(prediction, response_content_type):
    """After invoking predict_fn, the model server invokes `output_fn`.

    Serialize the prediction result into the desired response content type.
    """
    if response_content_type == "application/x-npy":
        np_bytes = BytesIO()
        np.save(np_bytes, prediction, allow_pickle=True)
        return np_bytes
    else:
        raise ValueError("Content type {} is not supported.".format(response_content_type))
