import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "window",
        help="Window size (if any) to manipulate training data",
        type=int,
        default=0,
    )
    return parser
