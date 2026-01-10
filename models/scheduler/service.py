from .core import SchedulingProblem, ScenarioRunner, Solution
from .fairness import FairnessStrategy
from .objectives import ObjectivePolicy

class SchedulerService:
    """
    Thin stateless wrapper around ScenarioRunner.
    Use the singleton factory in Streamlit (e.g. @st.cache_resource).
    """
    def __init__(self):
        pass

    def run(self, problem: SchedulingProblem, fairness: FairnessStrategy, objective: ObjectivePolicy) -> Solution:
        runner = ScenarioRunner(problem, fairness, objective)
        return runner.run()