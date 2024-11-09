class Metrics:
    def _compute_convergence(self, hint_score, up_to):
        if up_to == 1:
            scores = [hint_score[-1]]
        else:
            scores = hint_score[:up_to - 1] + [hint_score[-1]]
        if scores[-1] == 0:
            return 0
        return round(1 - ((sum(scores) - 1) / len(scores)), 2)

    def compute_metrics(self, scores):
        convergences = []
        for hint_score in scores:
            hint_score = list(hint_score.values())
            conv = self._compute_convergence(hint_score, len(hint_score))
            convergences.append(conv)
        return convergences
