from __future__ import annotations

import copy


class StandaloneOptimizerContext(dict):
    """Problem adapter used only by the legacy single-application API.

    Search engines receive a small explicit interface and never import one
    another or reimplement the scheduling/energy objective.
    """

    def __init__(self, base_context, *, seed=None):
        super().__init__(copy.deepcopy(dict(base_context)))
        if seed is not None:
            self["seed"] = int(seed)
        order = self.get("task_order") or self.get("ranked_task_ids")
        if not order:
            from algorithm.optimizer_common import compute_task_ranks_and_order

            order = compute_task_ranks_and_order(self)
        self["task_order"] = [int(value) for value in order]
        self.setdefault("initial_evaluation_memo", {})

    @property
    def task_order(self):
        return list(self["task_order"])

    @property
    def initial_function_evaluations(self):
        return 0

    def valid_provider(self, task):
        domains = self.get("task_domains", self.get("providers", {}))
        values = domains.get(int(task), []) if isinstance(domains, dict) else []
        if isinstance(values, dict):
            values = values.keys()
        return list(dict.fromkeys(int(value) for value in (values or [])))

    def repair_solution(self, solution):
        assignments = {}
        for gene in solution or []:
            if isinstance(gene, (tuple, list)) and len(gene) >= 2:
                assignments.setdefault(int(gene[0]), int(gene[1]))
            elif isinstance(gene, dict):
                task = gene.get("task")
                provider = gene.get("provider")
                if task is not None and provider is not None:
                    assignments.setdefault(int(task), int(provider))

        repaired = []
        for task in self.task_order:
            providers = self.valid_provider(task)
            if not providers:
                raise ValueError(f"Task {task} has no feasible provider")
            provider = int(assignments.get(task, providers[0]))
            if provider not in providers:
                provider = providers[0]
            repaired.append((int(task), provider, len(repaired)))
        return repaired

    def greedy_population(self, count):
        from algorithm.greedy_nests import procedure1_greedy_initialization

        return procedure1_greedy_initialization(
            S=max(1, int(count)),
            task_order=self.task_order,
            ctx=self,
        )

    def evaluate(self, solution):
        from algorithm.optimizer_common import evaluate_solution_quality

        result = evaluate_solution_quality(self, solution, self.task_order)
        return float(result[0] if isinstance(result, tuple) else result)

    def evaluate_solution(self, solution):
        return self.evaluate(solution)
