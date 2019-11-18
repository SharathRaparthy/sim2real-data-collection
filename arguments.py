import argparse

"""
Here are the parameters
"""
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--freq', type=int, default=1, help='the frequency by which recording is done')
    parser.add_argument('--approach', type=str, default='motor_babbling', help='the number of epochs to train the agent')
    parser.add_argument('--env-name', type=str, default='ErgoPusher', help='the number of epochs to train the agent')
    parser.add_argument('--action-noise', type=bool, default=False, help='the number of epochs to train the agent')
    parser.add_argument('--obs-noise', type=bool, default=False, help='the number of epochs to train the agent')
    parser.add_argument('--noise-type', type=str, default='action-noise', help='the number of epochs to train the agent')
    parser.add_argument('--variant', type=int, default=10, help='the number of epochs to train the agent')
    parser.add_argument('--task', type=str, default='pusher', help='the number of epochs to train the agent')
    parser.add_argument('--total-steps', type=int, default=int(1e6), help='the number of epochs to train the agent')
    parser.add_argument('--rest-interval', type=int, default=int(1e3), help='the number of epochs to train the agent')
    parser.add_argument('--num-steps', type=int, default=int(1e2), help='the number of epochs to train the agent')
    parser.add_argument('--history-len', type=int, default=int(1e4), help='the number of epochs to train the agent')
    parser.add_argument('--goal-sample-freq', type=int, default=1, help='the number of epochs to train the agent')
    parser.add_argument('--num-retries', type=int, default=5, help='the number of epochs to train the agent')
    parser.add_argument('--seed', type=int, default=123, help='the number of epochs to train the agent')

    args = parser.parse_args()

    return args