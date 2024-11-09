class Hint_Generation:
    def __init__(self, pipeline, num_of_hints, parse_llm_response, temperature, top_p, max_tokens):
        self.pipeline = pipeline
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

    def _execute_prompt(self, messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return [output[0]["generated_text"][-1]['content'].strip() for output in outputs]

    def _hint_prompt(self, pairs):
        messages = []
        for item in pairs:
            question, exact_answer = item
            message = [{"role": "system",
                        "content": "You are a helpful assistant that generates hints for user questions. You are given the question, and your goal is to generate hints for the question."},
                       {"role": "user",
                        "content": "Generate {} hints for the following question without using \"{}\" word in the hints. Question: {}".format(
                            self.num_of_hints, exact_answer, question)}]
            messages.append(message)
        outputs = self._execute_prompt(messages)
        results = dict()
        for pair, output in zip(pairs, outputs):
            results[pair[0]] = output
        question_hints_dict = {pairs[idx][0]: results[pairs[idx][0]] for idx in range(len(pairs))}
        return question_hints_dict

    def generate(self, pairs):
        question_hints_dict = self._hint_prompt(pairs)
        for q in question_hints_dict.keys():
            question_hints_dict[q] = self.parse_llm_response(question_hints_dict[q])
        return list(question_hints_dict.values())
