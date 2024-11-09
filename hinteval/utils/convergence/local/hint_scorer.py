class HintScorer:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @staticmethod
    def _clear_hint(hint: str):
        idx = hint.find('[^')
        if idx >= 0:
            hint = hint[:idx] + '.'
        return hint

    def _execute_prompt(self, messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=4,
            temperature=0.01,
            top_p=1.0
        )
        return [output[0]["generated_text"][-1]['content'].strip().lower() for output in outputs]

    def _hint_prompt(self, hint, candidate_answers):
        hint_candidate_list = list(zip([hint] * len(candidate_answers), candidate_answers))
        messages = []
        for item in hint_candidate_list:
            hint, can = item
            hint_prompt = f'Does the hint "{hint}" refer to "{can}"? Write ONLY between "Yes" or "No"'
            hint_prompt_message = [{"role": "user", "content": hint_prompt}]
            messages.append(hint_prompt_message)
        outputs_labels = self._execute_prompt(messages)
        results = dict()
        for can, label in zip(candidate_answers, outputs_labels):
            results[can] = 1 if label.lower().startswith('yes') else 0
        hint_candidate_answers_dict = {candidate_answers[idx]: results[candidate_answers[idx]] for idx in
                                       range(len(candidate_answers))}
        return hint_candidate_answers_dict

    def rate(self, hints, candidate_answers):
        scores = []
        for hint_idx, hint in enumerate(hints, start=1):
            hint = self._clear_hint(hint)
            hint_prompt = self._hint_prompt(hint, candidate_answers)
            scores.append(hint_prompt)
        return scores
