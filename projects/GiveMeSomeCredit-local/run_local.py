import os
import numpy as np
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

ROLE = "local-role"               # dummy for Local Mode
FRAMEWORK_VERSION = "1.2-1"

def main():
    # Put all local artifacts under ../../outputs
    local_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))
    os.makedirs(local_out, exist_ok=True)

    sm_local = LocalSession()
    sm_local.config = {"local": {"local_code": True}}

    est = SKLearn(
        entry_point="src/train.py",
        role=ROLE,
        instance_type="local",
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=sm_local,
        output_path=f"file://{local_out}"
    )

    est.fit()  # train (data is internal to train.py)

    predictor = est.deploy(initial_instance_count=1, instance_type="local")
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    print("Prediction:", predictor.predict(sample))

    predictor.delete_endpoint()  # stop local container

if __name__ == "__main__":
    main()
