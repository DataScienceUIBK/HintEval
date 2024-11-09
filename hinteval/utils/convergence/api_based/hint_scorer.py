import asyncio
from openai import AsyncOpenAI


class HintScorer:
    def __init__(self, base_url, api_key, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _clear_hint(hint: str):
        idx = hint.find('[^')
        if idx >= 0:
            hint = hint[:idx] + '.'
        return hint

    async def _execute_prompt(self, messages):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=512
        )
        return response

    async def _hint_thread(self, info):
        try:
            hint, can = info
            hint_prompt = f'Does the hint "{hint}" refer to "{can}"? Write ONLY between "Yes" or "No"'
            hint_prompt_messages = [{"role": "user", "content": hint_prompt}]
            hint_prompt_executed = await self._execute_prompt(hint_prompt_messages)
            inclusion = hint_prompt_executed.choices[0].message.content.strip().lower()
            inclusion = 1 if inclusion.lower().startswith('yes') else 0
            return can, inclusion
        except:
            return can, -1

    async def _hint_prompt(self, hint, candidate_answers):
        hint_candidate_list = list(zip([hint] * len(candidate_answers), candidate_answers))
        tasks = []
        for hint_candidate in hint_candidate_list:
            tasks.append(asyncio.create_task(self._hint_thread(hint_candidate)))
        results = dict()
        for task in asyncio.as_completed(tasks):
            task = await task
            results[task[0]] = task[1]
        hint_candidate_answers_dict = {candidate_answers[idx]: results[candidate_answers[idx]] for idx in
                                       range(len(candidate_answers))}
        return hint_candidate_answers_dict

    async def rate(self, hints, candidate_answers):
        scores = []
        for hint_idx, hint in enumerate(hints, start=1):
            hint = self._clear_hint(hint)
            hint_prompt = await self._hint_prompt(hint, candidate_answers)
            if -1 in hint_prompt.values():
                raise Exception()
            scores.append(hint_prompt)
        return scores
