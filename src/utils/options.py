"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    parser.add_argument('--f')
    # Central:
    parser.add_argument('--model', default='codeLlama', type=str)
    parser.add_argument('--close_model', default='claude', type=str)
    parser.add_argument('--setting', default='buggy_COT', type=str)
    parser.add_argument('--session', default=1, type=int)

    return parser