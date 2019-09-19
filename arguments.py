import argparse

"""
Here are the parameters
"""
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--freq', type=int, default=1, help='the frequency by which recording is done')
    parser.add_argument('--num-steps', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--approach', type=str, default='motor_babbling', help='the number of epochs to train the agent')

    args = parser.parse_args()

    return args