import os
import argparse
import logging

import train
import test
import infer


__description__ = "Project Launcher Script"

SUB_PROCEDURES = {
    'train': train.main,
    'test': test.main,
    'infer': infer.main
}

def main(args):
    """
    Main script entry-point.

    :param args: parse-args arguments.
    :return: 0 if success.
    """

    if not args.__contains__('sub_procedure') or args.sub_procedure not in SUB_PROCEDURES:
        raise RuntimeError("Please specify a valid sub-procedure to run.")
    else:
        return SUB_PROCEDURES[args.sub_procedure](args)


def init_args(parser):
    """
    Initializes the argparse arguments, describes program usage.

    :param parser: ArgumentParser to initialize with arguments.
    :return: initialized ArgumentParser
    """
    sub_parsers = parser.add_subparsers(help="Project Procedures")

    train_parser = train.init_args(sub_parsers.add_parser('train', help="TODO"))
    train_parser.set_defaults(sub_procedure='train')

    test_parser = test.init_args(sub_parsers.add_parser('test', help="TODO"))
    test_parser.set_defaults(sub_procedure='test')

    infer_parser = infer.init_args(sub_parsers.add_parser('infer', help="TODO"))
    infer_parser.set_defaults(sub_procedure='infer')
    return parser


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    logging.info(f"Starting {__file__}")
    logging.info(f"PYTHONPATH=\"{os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else ''}\"")
    logging.info(f"{sys.executable} {' '.join(sys.argv)}")
    exit(main(init_args(argparse.ArgumentParser(description=__description__)).parse_args()))
