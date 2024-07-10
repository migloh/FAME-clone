from utils.config import get_config
from solver.moesolver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option_path', type=str, default='src\option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    print(cfg)