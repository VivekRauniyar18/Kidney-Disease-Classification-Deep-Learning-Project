from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainigPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline




STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>> stage{STAGE_NAME} started <<<<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e  


STAGE_NAME = "Prepare base model"
try:
    logger.info(f"*******************************")
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    obj = PrepareBaseModelTrainigPipeline()
    obj.main()
    logger.info(f">>>>>>>> stae { STAGE_NAME} completed <<<<<,\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"
try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation stage"
try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<")
    model_evaluation = ModelTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e