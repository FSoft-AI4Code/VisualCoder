"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    parser.add_argument('--f')
    # Central:
    parser.add_argument('--close_model', default='claude', type=str)
    parser.add_argument('--setting', default='buggy_COT', type=str)
    parser.add_argument('--session', default="1", type=str)
    parser.add_argument('--claude_api_key', default=None, type=str)
    parser.add_argument('--openai_api_key', default=None, type=str)
    parser.add_argument('--azure_endpoint', default=None, type=str)
    parser.add_argument('--deployment_name', default=None, type=str)
    parser.add_argument('--version', default='claude-3-5-sonnet-20240620', type=str)
    parser.add_argument('--prompt_mode', default='zeroshot', type=str)

    return parser