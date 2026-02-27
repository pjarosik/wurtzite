import argparse
import pickle


def read_default_params():
    parser = argparse.ArgumentParser(description="Wurtzite runtime")

    parser.add_argument(
        "--init",
        type=str,
        required=False,
        default=None,
        help="The path to the initial state file (.pkl)."
    )
    parser.add_argument(
        "--skip_save_state",
        action="store_true",
        default=False,
        help="Turns off saving the current state to the .pkl file"
    )
    parser.add_argument(
        "--start",
        type=int,
        required=False,
        default=None,
        help="The dislocation from which the script should start. NOTE: this "
             "parameters should be used along with the --init parameter."
    )

    args = parser.parse_args()
    init_state = None
    if args.init:
        init_state = pickle.load(open(args.init, "rb"))
        start_dislocation = args.start

    return {
        "init_state": init_state,
        "skip_save_state": args.skip_save_state,
        "start_dislocation": args.start-1 if args.start is not None else None
    }
