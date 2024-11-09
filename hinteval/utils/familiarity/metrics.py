class Metrics:

    def _compute_familiarity(self, normalized_pops):
        if len(normalized_pops) == 0:
            return 1.0
        avg_pop = round(sum(normalized_pops) / len(normalized_pops), 3)
        return avg_pop

    def compute_metrics(self, scores):
        normalized_pops = list(scores.values())
        familiarity = self._compute_familiarity(normalized_pops)
        return familiarity
