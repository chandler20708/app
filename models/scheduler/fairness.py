from .core import FairnessStrategy, SchedulingProblem
import gurobipy as gp
import itertools

# --- Strategy Implementations ---
class NoFairness(FairnessStrategy):
    """
    Baseline strategy with no fairness constraints.
    """
    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem): return None

class AllPairsFairness(FairnessStrategy):
    """
    Implements all-pairs fairness by minimising the sum of absolute
    differences in weekly workload between every pair of students.
    """
    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):
        pairs = itertools.combinations(problem.students, 2)
        diff_vars = []
        for (o1, o2) in pairs:
            d_var = model.addVar(lb=0, name=f"diff_{o1}_{o2}")
            h1 = gp.quicksum(H[d, o1] for d in problem.days)
            h2 = gp.quicksum(H[d, o2] for d in problem.days)
            model.addConstr(d_var >= h1 - h2, name=f"FairDiff_Pos[{o1},{o2}]")
            model.addConstr(d_var >= h2 - h1, name=f"FairDiff_Neg[{o1},{o2}]")
            diff_vars.append(d_var)
        return gp.quicksum(diff_vars)
    
class AllPairsAvailabilityFairness(FairnessStrategy):
    """
    All-pairs fairness on availability-normalised weekly workloads.

    Minimises the sum of absolute differences between
    H_o / Avail_o^{max} across all student pairs.
    """

    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):
        students = problem.students
        days = problem.days

        # Weekly hours
        weekly = {
            o: gp.quicksum(H[d, o] for d in days)
            for o in students
        }

        # Max weekly availability
        weekly_avail = {
            o: sum(problem.availability[d, o] for d in days)
            for o in students
        }

        diff_vars = []

        for o1, o2 in itertools.combinations(students, 2):
            d_var = model.addVar(lb=0, name=f"UtilDiff[{o1},{o2}]")

            util_1 = weekly[o1] / weekly_avail[o1]
            util_2 = weekly[o2] / weekly_avail[o2]

            model.addConstr(d_var >= util_1 - util_2, name=f"UtilDiff_Pos[{o1},{o2}]")
            model.addConstr(d_var >= util_2 - util_1, name=f"UtilDiff_Neg[{o1},{o2}]")

            diff_vars.append(d_var)

        return gp.quicksum(diff_vars)

class LexicographicFairness(FairnessStrategy):
    """
    Constructs lexicographic fairness components:
    1. Minimise maximum deviation from mean workload (L∞ fairness)
    2. Minimise range (max − min workload)
    """
    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):
        ops = problem.students
        weekly = {o: gp.quicksum(H[d, o] for d in problem.days) for o in ops}
        
        # Mean & Deviation
        mean = model.addVar(lb=0, name="mean_hours")
        model.addConstr(mean * len(ops) == gp.quicksum(weekly.values()), name="Mean_Workload_Def")
        
        D = model.addVar(lb=0, name="max_deviation")
        for o in ops:
            model.addConstr(weekly[o] - mean <= D, name=f"Dev_Pos[{o}]")
            model.addConstr(mean - weekly[o] <= D, name=f"Dev_Neg[{o}]")
            
        # Range
        h_max = model.addVar(lb=0, name="h_max")
        h_min = model.addVar(lb=0, name="h_min")
        for o in ops:
            model.addConstr(h_min <= weekly[o], name=f"Range_Min[{o}]")
            model.addConstr(weekly[o] <= h_max, name=f"Range_Max[{o}]")
            
        return {
            "mean": mean,
            "D": D,
            "h_min": h_min,
            "h_max": h_max,
            "range": h_max - h_min
        }
    
class LexicographicAvailabilityFairness(FairnessStrategy):
    """
    Lexicographic fairness on availability-normalised workloads.

    Priority 1: minimise maximum deviation from mean utilisation (L∞)
    Priority 2: minimise utilisation range
    """

    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):
        students = problem.students
        days = problem.days

        # Weekly hours
        weekly = {
            o: gp.quicksum(H[d, o] for d in days)
            for o in students
        }

        # Max weekly availability
        weekly_avail = {
            o: sum(problem.availability[d, o] for d in days)
            for o in students
        }

        # Normalised utilisation
        util = {
            o: weekly[o] / weekly_avail[o]
            for o in students
        }

        # Mean utilisation
        mean_util = model.addVar(lb=0, name="mean_utilisation")
        model.addConstr(
            mean_util * len(students) == gp.quicksum(util.values()),
            name="Mean_Utilisation_Def"
        )

        # Max deviation
        D = model.addVar(lb=0, name="max_util_deviation")
        for o in students:
            model.addConstr(util[o] - mean_util <= D, name=f"UtilDev_Pos[{o}]")
            model.addConstr(mean_util - util[o] <= D, name=f"UtilDev_Neg[{o}]")

        # Utilisation range
        u_max = model.addVar(lb=0, name="util_max")
        u_min = model.addVar(lb=0, name="util_min")
        for o in students:
            model.addConstr(u_min <= util[o], name=f"UtilRange_Min[{o}]")
            model.addConstr(util[o] <= u_max, name=f"UtilRange_Max[{o}]")

        # utilisation parameter
        total_required_hours = problem.daily_coverage_req * len(problem.days)
        total_available_hours = sum(weekly_avail.values())
        alpha = total_required_hours / total_available_hours

        U = {}

        for o in students:
            target_o = alpha * weekly_avail[o]
            U[o] = model.addVar(lb=0, name=f"UnderAlloc[{o}]")
            model.addConstr(
                U[o] >= target_o - weekly[o],
                name=f"LowerTail[{o}]"
            )

        return {
            "mean": mean_util,
            "D": D,
            "u_min": u_min,
            "u_max": u_max,
            "range": u_max - u_min,
            "lower_tail": gp.quicksum(U[o] for o in students)
        }
    
class AvailabilityLowerTailFairness(FairnessStrategy):
    """
    Availability-weighted lower-tail fairness (one-sided MAE).

    Penalises only under-allocation relative to an availability-based
    target:
        t_o = alpha * max_weekly_availability_o
    """

    def apply(self, model: gp.Model, H: gp.tupledict, problem: SchedulingProblem):

        weekly_hours = {}
        weekly_avail = {}

        for o in problem.students:
            weekly_hours[o] = gp.quicksum(H[d, o] for d in problem.days)
            weekly_avail[o] = sum(problem.availability[d, o] for d in problem.days)

        # utilisation parameter
        total_required_hours = problem.daily_coverage_req * len(problem.days)
        total_available_hours = sum(weekly_avail.values())
        alpha = total_required_hours / total_available_hours

        U = {}  # under-allocation

        for o in problem.students:
            target = alpha * weekly_avail[o]
            U[o] = model.addVar(lb=0, name=f"UnderAlloc[{o}]")

            model.addConstr(
                U[o] >= target - weekly_hours[o],
                name=f"LowerTailFairness[{o}]"
            )

        return gp.quicksum(U[o] for o in problem.students)

        # U = {}  # under-allocation
        # O = {}  # over-allocation

        # lambda_over = 0.2   # mild penalty on over-allocation
        # beta = 1.25         # soft upper bound multiple

        # for o in problem.students:
        #     target = alpha * avail_max[o]

        #     U[o] = model.addVar(lb=0, name=f"UnderAlloc[{o}]")
        #     O[o] = model.addVar(lb=0, name=f"OverAlloc[{o}]")

        #     model.addConstr(
        #         U[o] >= target - weekly_hours[o],
        #         name=f"LowerTail_Under[{o}]"
        #     )

        #     model.addConstr(
        #         O[o] >= weekly_hours[o] - beta * target,
        #         name=f"LowerTail_Over[{o}]"
        #     )

        # return gp.quicksum(
        #     U[o] + lambda_over * O[o]
        #     for o in problem.students
        # )
