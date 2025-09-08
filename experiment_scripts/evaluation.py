from bank_marketing_term_deposit_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
