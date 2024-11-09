import outlines
from pydantic import BaseModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM


class _CustomModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    async def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        generator = outlines.generate.json(model, schema)
        resp = generator(prompt)
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return await self.generate(prompt, schema)

    def get_model_name(self):
        return "CustomModel"


class AnswerRelevancy:
    def __init__(self, model_name, threshold):
        self.model = outlines.models.transformers(model_name, model_kwargs={"device_map": "auto"})
        self.threshold = threshold

    def _compute_similarity(self, question, hints):
        metric = AnswerRelevancyMetric(
            model=_CustomModel(self.model),
            threshold=self.threshold,
            include_reason=False
        )
        test_cases = [LLMTestCase(input=question, actual_output=hint) for hint in hints]

        results = []
        for test_case in test_cases:
            metric.measure(test_case, _show_indicator=False)
            results.append(metric.score)

        return results

    def compute_relevancy(self, question, hints):
        return self._compute_similarity(question, hints)
