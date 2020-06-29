import argparse

class DataHparams:
    parser = argparse.ArgumentParser()
    parser.add_argument('--thchs30', default=False, type=bool)
    parser.add_argument('--aishell', default=True, type=bool)
    parser.add_argument('--prime', default=False, type=bool)
    parser.add_argument('--stcmd', default=False, type=bool)
    parser.add_argument('--aidatatang', default=False, type=bool)
    parser.add_argument('--magic_data', default=False, type=bool)
    parser.add_argument('--aidatatang_1505', default=False, type=bool)

    parser.add_argument('--bgn_data', default=False, type=bool)
    parser.add_argument('--echo_data', default=False, type=bool)
    parser.add_argument('--noise_data', default=False, type=bool)
    parser.add_argument('--rate_data', default=False, type=bool)

    parser.add_argument('--train_length', default=None, type=int)