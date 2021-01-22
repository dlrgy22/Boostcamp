import argparse

parser = argparse.ArgumentParser()        # extra value
parser.add_argument("-f", "--fast", dest="fast", action="store_true")           # existence/nonexistence
args = parser.parse_args()

if args.fast:
    print("-f option is used")
