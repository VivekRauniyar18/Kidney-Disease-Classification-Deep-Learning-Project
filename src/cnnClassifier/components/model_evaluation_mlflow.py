import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"

        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            shuffle = False,
            subset = "validation",
            
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluate(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path = Path("scores.json"), data=scores)

    

    def log_into_mlflow(self):
        import os
        import shutil
        from pathlib import Path
        import mlflow
        from mlflow.models import infer_signature

        # --- 1) Point MLflow to DagsHub in-code (works even if env vars aren't loaded) ---
        DAGSHUB_USER = "VivekRauniyar18"
        DAGSHUB_REPO = "Kidney-Disease-Classification-Deep-Learning-Project"  # <- double-check repo name
        MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"

        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", DAGSHUB_USER)
        # If you prefer a token:
        # os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", "<your_token_here>")
        # (If you already export these in the shell, this won't overwrite them.)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_registry_uri(MLFLOW_URI)  # harmless; registry ops will be skipped

        # Optional: organize runs
        mlflow.set_experiment(getattr(self.config, "mlflow_experiment_name", "kidney-eval"))

        # --- 2) Start run and log params/metrics ---
        with mlflow.start_run() as run:
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": float(self.score[0]), "accuracy": float(self.score[1])})

            # --- 3) Add a signature to silence the warning ---
            # grab a small batch from the validation generator
            x_batch, _ = next(self.valid_generator)
            preds = self.model.predict(x_batch, verbose=0)
            signature = infer_signature(x_batch, preds)
            input_example = x_batch[:2]  # tiny example

            # --- 4) Save as TF SavedModel and log as artifacts (works with Keras 3) ---
            export_dir = Path("tmp_saved_model")
            if export_dir.exists():
                shutil.rmtree(export_dir)
            self.model.save(export_dir.as_posix())  # Keras SavedModel

            # Log the folder (generic artifacts)
            mlflow.log_artifacts(export_dir.as_posix(), artifact_path="model")

            # Also log via the TensorFlow flavor WITHOUT registry (handles signature)
            try:
                import mlflow.tensorflow as mlt
                mlt.log_model(
                    self.model,
                    artifact_path="tf_flavor_model",
                    signature=signature,
                    input_example=input_example,
                    # Do NOT pass registered_model_name on DagsHub (registry not supported)
                )
            except Exception as e:
                print("[MLflow] TF flavor log_model skipped:", e)

            print("[MLflow] tracking_uri:", mlflow.get_tracking_uri())
            print("[MLflow] run_id:", run.info.run_id)
