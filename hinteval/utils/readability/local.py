import transformers
import re


class ReadmeReadability:
    def __init__(self, model_name, temperature, top_p, max_tokens):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id
        self.pattern = r'\d+'
        self.system_prompt = """
        Rate the following sentence on it’s readability level. The readability is defined
        as the cognitive load required to understand the meaning of the sentence. Rate
        the readability on a scale from very easy to very hard. Base your scores off the
        CEFR scale for L2 Learners. You should use the following key:

        1 = Can understand very short, simple texts a single phrase at a time, picking up
        familiar names, words and basic phrases and rereading as required.
        2 = Can understand short, simple texts on familiar matters of a concrete type
        3 = Can read straightforward factual texts on subjects related to his/her field
        and interest with a satisfactory level of comprehension.
        4 = Can read with a large degree of independence, adapting style and speed of
        reading to different texts and purpose
        5 = Can understand in detail lengthy, complex texts, whether or not they relate
        to his/her own area of speciality, provided he/she can reread difficult sections.
        6 = Can understand and interpret critically virtually all forms of the written
        language including abstract, structurally complex, or highly colloquial literary
        and non-literary writings.
        """

        self.examples = [
            (
                'Sentence: "They took over China, Persia, Turkestan, and Russia."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '1'),
            (
                'Sentence: "Heavy smokers are reported to burn 200 calories per day more than non-smokers eating the same diet."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '2'),
            (
                'Sentence: "However, even using these improved methods, the total number of bacterial species is not known and cannot even be estimated with any certainty."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '3'),
            (
                'Sentence: "It increases in circumstances where someone has reduced immunity to infection."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '4'),
            (
                'Sentence: "Counselling Counselling psychology is a psychology specialty that facilitates personal and interpersonal functioning across the lifespan with a focus on emotional, social, vocational, educational, health- related, developmental, and organizational concerns."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '5'),
            (
                'Sentence: "Eukaryotic cells also have cytoskeleton that is made up of microtubules, intermediate filaments, and microfilaments, all of which provide support for the cell and are involved in the movement of the cell and its organelles."\nGiven the above key, the readability of the sentence is (scale=1-6): ',
                '6')
        ]

    def _compute_by_prompting(self, sentences):
        results = []
        for sentence in sentences:
            messages = [{"role": "system", "content": self.system_prompt}]
            for example in self.examples:
                messages.append({"role": "user", "content": example[0]})
                messages.append({"role": "assistant", "content": example[1]})
            messages.append({"role": "user",
                             "content": f'Sentence: "{sentence}"\nGiven the above key, the readability of the sentence is (scale=1-6): '})
            outputs = self.pipeline(messages)
            response = outputs[0]["generated_text"][-1]['content'].strip()
            match = re.search(self.pattern, response)
            if match:
                number = abs((int(match.group(0)) - 1) // 2)
                if number <= 2:
                    results.append(number)
                else:
                    results.append(1)
            else:
                results.append(1)
        return results

    def compute_readability(self, sentences):
        return self._compute_by_prompting(sentences)
