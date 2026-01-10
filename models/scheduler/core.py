import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# Data Structures
@dataclass(frozen=True)
class SchedulingProblem:
    """
    Immutable container defining a scheduling optimisation instance.

    Attributes
    ----------
    students : list[str]
        Canonical ordered list of student operators.
    days : list[str]
        Canonical ordered list of scheduling days.
    availability : gp.tupledict[(str, str), float]
        Maximum available hours for each (day, student) pair.
    wage_rates : dict[str, float]
        Hourly wage rate per student.
    bachelor_students : set[str]
        Subset of students enrolled in bachelor programmes.
    master_students : set[str]
        Subset of students enrolled in master programmes.
    programming_ops : list[str]
        Students qualified for programming tasks (optional).
    troubleshooting_ops : list[str]
        Students qualified for troubleshooting tasks (optional).
    daily_coverage_req : float
        Required total staffing hours per day.
    """
    students: list[str]
    days: list[str]
    availability: gp.tupledict[tuple[str, str], int | float]
    wage_rates: dict[str, int | float]
    bachelor_students: set[str]
    master_students: set[str]
    programming_ops: list[str] = field(default_factory=list)
    troubleshooting_ops: list[str] = field(default_factory=list)
    daily_coverage_req: float = 14.0

@dataclass
class Solution:
    """
    Container holding the result of an optimisation run.

    Attributes
    ----------
    H_vars : gp.tupledict
        Decision variables representing allocated hours.
    weekly_hours : dict[str, float]
        Total weekly hours worked per student.
    total_cost : float
        Total labour cost of the solution.
    model : gp.Model
        Solved Gurobi model instance.
    problem : SchedulingProblem
        Reference to the originating problem definition.
    """
    H_vars: gp.tupledict
    weekly_hours: dict[str, int | float]
    total_cost: float
    model: gp.Model
    problem: SchedulingProblem

# --- Strategy Interfaces ---
class FairnessStrategy(ABC):
    """
    Abstract interface for fairness constraint generation.
    """
    @abstractmethod
    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):
        """Adds fairness variables/constraints and returns the primary fairness expression."""
        pass

class ObjectivePolicy(ABC):
    """
    Abstract interface for objective-function configuration.
    """
    @abstractmethod
    def set_objective(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem, fairness_expr=None):
        pass
    
# --- Result Display ---
class VisualReporter(ABC):
    """
    Abstract interface for solution reporting.
    """
    @abstractmethod
    def report(self, solution: Solution, title: str): pass


# --- Model Builder & Runner ---

class ModelBuilder:
    def __init__(self, problem: SchedulingProblem):
        self.problem = problem

    def build_core(self) -> tuple[gp.Model, gp.tupledict]:
        m = gp.Model("Scheduler")
        H = m.addVars(self.problem.days, self.problem.students, lb=0, ub=self.problem.availability, name="H")

        # Constraints: Minimum weekly commitments for bachelor students
        m.addConstrs(
            (gp.quicksum(H[d, o] for d in self.problem.days) >= 8 for o in self.problem.bachelor_students),
            name='Bachelor_Min'
        )

        # Constraints: Minimum weekly commitments for master students
        m.addConstrs(
            (gp.quicksum(H[d, o] for d in self.problem.days) >= 7 for o in self.problem.master_students),
            name='Master_Min'
        )

        # Constraints: Mainframe operating hours per day
        m.addConstrs(
            (gp.quicksum(H[d, o] for o in self.problem.students) == self.problem.daily_coverage_req for d in self.problem.days),
            name='Daily_Coverage'
        )

        # Optional Skill Constraints
        if self.problem.programming_ops:
            # Ensure at least 6 hours of troubleshooting skill per day
            # We sum the hours of everyone who has the troubleshooting skill
            m.addConstrs(
                (gp.quicksum(H[d, o] for o in self.problem.programming_ops) >= 6 for d in self.problem.days),
                name='Min_Programming_Hours'
            )

        if self.problem.troubleshooting_ops:
            # Ensure at least 6 hours of troubleshooting skill per day
            # We sum the hours of everyone who has the troubleshooting skill
            m.addConstrs(
                (gp.quicksum(H[d, o] for o in self.problem.troubleshooting_ops) >= 6 for d in self.problem.days),
                name='Min_Troubleshooting_Hours'
            )

        return m, H

class ScenarioRunner:
    def __init__(self, problem: SchedulingProblem, fairness: FairnessStrategy, objective: ObjectivePolicy):
        self.problem = problem
        self.fairness = fairness
        self.objective = objective

    def run(self) -> Solution:
        builder = ModelBuilder(self.problem)
        model, H = builder.build_core()
        
        fairness_expr = self.fairness.apply(model, H, self.problem)
        self.objective.set_objective(model, H, self.problem, fairness_expr)
        
        model.optimize()
        
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("Optimal solution not found.")

        # Extract results
        weekly = {s: sum(H[d, s].X for d in self.problem.days) for s in self.problem.students}
        cost = sum(H[d, s].X * self.problem.wage_rates[s] for d, s in H.keys())
        
        return Solution(H, weekly, cost, model, self.problem)
    