from .core import ObjectivePolicy, SchedulingProblem
from typing import Optional
from gurobipy import GRB
import gurobipy as gp


class MinimizeCostPolicy(ObjectivePolicy):
    """
    Single-objective policy that minimises total labour cost.
    """

    def set_objective(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem, fairness_expr: Optional[gp.LinExpr]=None):
        cost_expr = gp.quicksum(H[d, s] * problem.wage_rates[s] 
                                for d in problem.days for s in problem.students)
        model.setObjective(cost_expr, GRB.MINIMIZE)

class MinimizeUnfairnessPolicy(ObjectivePolicy):
    """
    Minimises a fairness expression, optionally subject to a cost cap.
    """
    def __init__(self, cost_limit: Optional[float] = None):
        self.cost_limit = cost_limit

    def set_objective(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem, fairness_expr: Optional[gp.LinExpr]=None):
        if self.cost_limit is not None:
            cost_expr = gp.quicksum(H[d, s] * problem.wage_rates[s] 
                                    for d in problem.days for s in problem.students)
            model.addConstr(cost_expr <= self.cost_limit, name="BudgetConstraint")
        
        if fairness_expr is not None:
            model.setObjective(fairness_expr, GRB.MINIMIZE)

class LexicographicObjectivePolicy(ObjectivePolicy):
    """
    Applies lexicographic (hierarchical) optimisation objectives.
    """

    def __init__(self, cost_limit: Optional[float] = None):
        self.cost_limit = cost_limit

    def set_objective(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem, fairness_expr: Optional[dict[str, float | int | gp.LinExpr]] = None):
        if self.cost_limit is not None:
            cost = gp.quicksum(H[d, s] * problem.wage_rates[s] for d, s in H.keys())
            model.addConstr(cost <= self.cost_limit, name="Budget")
        
        # Multi-objective: Priority 2 for Max Deviation, Priority 1 for Range
        assert fairness_expr is not None, f"Please input 'fairness_expr' for {self.__class__.__name__}"
        model.setObjectiveN(fairness_expr["D"], index=0, priority=5, name="MinDev")
        model.setObjectiveN(fairness_expr["range"], index=1, priority=10, name="MinRange")

class PriorityWeightedFairnessPolicy(ObjectivePolicy):
    """
    Single-objective approximation of lexicographic fairness
    using priority-weighted aggregation.
    """

    def __init__(
        self,
        w_dev: float = 20.0,
        w_range: float = 10.0,
        w_lower_tail: float = 1.0,
        cost_limit: float | None = None,
    ):
        self.w_dev = w_dev
        self.w_range = w_range
        self.w_lower_tail = w_lower_tail
        self.cost_limit = cost_limit

    def set_objective(
        self,
        model: gp.Model,
        H: gp.tupledict,
        problem: SchedulingProblem,
        fairness_expr: Optional[dict[str, float | int | gp.LinExpr]] = None
    ):
        if self.cost_limit is not None:
            cost = gp.quicksum(
                H[d, s] * problem.wage_rates[s] for d, s in H.keys()
            )
            model.addConstr(cost <= self.cost_limit, name="Budget")

        assert fairness_expr is not None, f"Please input 'fairness_expr' for {self.__class__.__name__}"
        model.setObjective(
            self.w_dev   * fairness_expr["D"]
          + self.w_range * fairness_expr["range"]
          + self.w_lower_tail * fairness_expr["lower_tail"],
            GRB.MINIMIZE
        )
