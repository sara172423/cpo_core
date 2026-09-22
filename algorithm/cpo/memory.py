from __future__ import annotations

from collections import defaultdict
import math


DEFENSE_NAMES = ("sight", "sound", "odor", "physical_attack")


class DefenseSuccessMemory:
    """Bounded accepted-move memory used only to guide future proposals.

    It never modifies the objective value.  Strategy credit is assigned only
    when a child replaces its parent, and provider credit is assigned only to
    coordinates changed by that accepted child.
    """

    def __init__(self, context, *, evaporation=0.10, provider_weight=0.65):
        self.evaporation = max(0.0, min(0.95, float(evaporation)))
        self.provider_weight = max(0.0, min(1.0, float(provider_weight)))
        self.task_types = context.get("task_type_ids", {}) or {}
        self.strategy_credit = {name: 1.0 for name in DEFENSE_NAMES}
        self.task_provider = defaultdict(float)
        self.service_provider = defaultdict(float)
        self.accepted_by_strategy = defaultdict(int)

    def begin_generation(self):
        factor = 1.0 - self.evaporation
        for name in DEFENSE_NAMES:
            self.strategy_credit[name] = max(
                0.05, float(self.strategy_credit.get(name, 1.0)) * factor
            )
        for table in (self.task_provider, self.service_provider):
            for key in list(table):
                table[key] *= factor
                if table[key] < 1e-10:
                    del table[key]

    def _task_type(self, task):
        value = self.task_types.get(int(task), self.task_types.get(str(int(task))))
        return None if value is None else int(value)

    @staticmethod
    def _normalized(values):
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high <= low:
            return {key: (1.0 if high > 0.0 else 0.0) for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    def reward(self, strategy, child, parent, gain):
        gain = max(0.0, float(gain))
        if gain <= 0.0:
            return False
        strategy = str(strategy)
        scaled = min(2.0, 0.25 + math.log1p(gain))
        self.strategy_credit[strategy] = self.strategy_credit.get(strategy, 0.05) + scaled
        self.accepted_by_strategy[strategy] += 1

        before = {
            int(gene[0]): int(gene[1])
            for gene in parent or []
            if isinstance(gene, (tuple, list)) and len(gene) >= 2
        }
        changed = [
            (int(gene[0]), int(gene[1]))
            for gene in child or []
            if isinstance(gene, (tuple, list))
            and len(gene) >= 2
            and before.get(int(gene[0])) != int(gene[1])
        ]
        if not changed:
            return False
        deposit = scaled / float(len(changed))
        for task, provider in changed:
            self.task_provider[(task, provider)] += deposit
            task_type = self._task_type(task)
            if task_type is not None:
                self.service_provider[(task_type, provider)] += deposit
        return True

    def strategy_weights(self, names):
        values = {
            str(name): max(0.05, float(self.strategy_credit.get(str(name), 0.05)))
            for name in names
        }
        total = sum(values.values()) or 1.0
        return {name: value / total for name, value in values.items()}

    def provider_scores(self, task, providers):
        providers = list(dict.fromkeys(int(value) for value in providers))
        direct = self._normalized(
            {provider: self.task_provider.get((int(task), provider), 0.0) for provider in providers}
        )
        task_type = self._task_type(task)
        shared = self._normalized(
            {
                provider: self.service_provider.get((task_type, provider), 0.0)
                if task_type is not None
                else 0.0
                for provider in providers
            }
        )
        return {
            provider: self.provider_weight * direct.get(provider, 0.0)
            + (1.0 - self.provider_weight) * shared.get(provider, 0.0)
            for provider in providers
        }

    def profile(self):
        return {
            "cpo_strategy_credit": {
                name: float(self.strategy_credit.get(name, 0.0))
                for name in DEFENSE_NAMES
            },
            "cpo_accepted_by_strategy": {
                name: int(self.accepted_by_strategy.get(name, 0))
                for name in DEFENSE_NAMES
            },
            "cpo_task_provider_memory_entries": int(len(self.task_provider)),
            "cpo_service_provider_memory_entries": int(len(self.service_provider)),
        }
