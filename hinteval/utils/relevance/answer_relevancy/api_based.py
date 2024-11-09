import instructor
from pydantic import BaseModel
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


class _CustomModel(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def load_model(self):
        return self.client

    async def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_openai(
            client=client,
            mode=instructor.Mode.JSON
        )
        resp = instructor_client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return await self.generate(prompt, schema)

    def get_model_name(self):
        return "CustomModel"


class AnswerRelevancy:
    def __init__(self, model_name, api_key, base_url, threshold):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.threshold = threshold

    def _compute_similarity(self, question, hints):
        metric = AnswerRelevancyMetric(
            model=_CustomModel(self.model_name, self.api_key, self.base_url),
            threshold=self.threshold, include_reason=False)
        test_cases = [LLMTestCase(input=question, actual_output=hint) for hint in hints]

        results = []
        for test_case in test_cases:
            metric.measure(test_case, _show_indicator=False)
            results.append(metric.score)
        return results

    def compute_relevancy(self, question, hints):
        return self._compute_similarity(question, hints)
