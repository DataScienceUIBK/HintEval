import asyncio
from openai import AsyncOpenAI


class Hint_Generation:
    def __init__(self, base_url, api_key, model_name, num_of_hints, parse_llm_response, temperature, top_p, max_tokens):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.num_of_hints = num_of_hints
        self.parse_llm_response = self._clear_candidate if parse_llm_response is None else parse_llm_response
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @staticmethod
    def _clear_candidate(prompt_content):
        candidate_raw: list = prompt_content.strip().split('\n')
        if len(candidate_raw) > 1 and candidate_raw[1] == '':
            cleared_candidate = candidate_raw[2:]
        else:
            cleared_candidate = candidate_raw

        if len(cleared_candidate) > 1 and cleared_candidate[-2] == '':
            cleared_candidate = cleared_candidate[:-2]
        else:
            cleared_candidate = cleared_candidate

        for i in range(len(cleared_candidate)):
            can = cleared_candidate[i]
            start_chars = ('-', '•', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')
            if can.strip().startswith(start_chars):
                if can.strip().startswith('10'):
                    can = can[3:].strip()
                else:
                    can = can[2:].strip()
            if can.strip().startswith('\"'):
                can = can.strip()[1:-1]
            cleared_candidate[i] = can
        return cleared_candidate

    async def _execute_prompt(self, messages):
        result = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        return result

    async def _hint_thread(self, pair):
        question, exact_answer = pair
        messages = [{"role": "system",
                     "content": "You are a helpful assistant that generates hints for user questions. You are given the question, and your goal is to generate hints for the question."},
                    {"role": "user",
                     "content": "Generate {} hints for the following question without using \"{}\" word in the hints. Question: {}".format(
                         self.num_of_hints, exact_answer, question)}]

        hint_prompt_executed = await self._execute_prompt(messages)
        hints = hint_prompt_executed.choices[0].message.content.strip()
        return question, hints

    async def _hint_prompt(self, pairs):
        tasks = []
        for question_answer in pairs:
            tasks.append(asyncio.create_task(self._hint_thread(question_answer)))
        results = dict()
        for task in asyncio.as_completed(tasks):
            task = await task
            results[task[0]] = task[1]
        question_hints_dict = {pairs[idx][0]: results[pairs[idx][0]] for idx in range(len(pairs))}
        return question_hints_dict

    async def generate(self, pairs):
        question_hints_dict = await self._hint_prompt(pairs)
        for q in question_hints_dict.keys():
            question_hints_dict[q] = self.parse_llm_response(question_hints_dict[q])
        return list(question_hints_dict.values())
