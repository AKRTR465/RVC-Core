import argparse
from src.index.cli import build_index, build_parser, resolve_manual_request, resolve_project_request


def parse_args():
    parser = build_parser(include_feature_dim=True, include_n_cpu=True)
    args = parser.parse_args()

    if args.config:
        if args.inp_root or args.output or args.feature_dim or args.n_cpu:
            parser.error("config mode only accepts --config, --hparams, and --reset")
        return resolve_project_request(args, include_n_cpu=True)

    if args.inp_root == "" or args.output == "" or args.feature_dim == 0:
        parser.error("manual mode requires --inp_root, --output, and --feature-dim")
    return resolve_manual_request(args, parser, include_n_cpu=True)


def main():
    request = parse_args()
    build_index(request)


if __name__ == "__main__":
    main()
