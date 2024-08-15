from utils.config  import get_config
from solver.testsolver import TestSolver
# from solver.midntestsolver import Testsolver
# from solver.inntestsolver import Testsolver
if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = TestSolver(cfg)
    solver.run()
    