import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="TextCNN sentiment and robustness pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download and persist shared data splits")
    p.add_argument("--output", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ood", choices=["yelp", "amazon", "none"], default="yelp")
    p.add_argument("--ood-limit", type=int, default=0)
    p.add_argument("--revision", default=None, help="Optional IMDB dataset commit revision")
    p = sub.add_parser("train")
    p.add_argument("--data", default="data")
    p.add_argument("--run", default="runs/textcnn_seed42")
    p.add_argument("--config", default="configs/baseline.json")
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="clean")
    p.add_argument("--device", default="auto")
    p = sub.add_parser("perturb")
    p.add_argument("--input", default="data/test.jsonl")
    p.add_argument("--output", default="data/perturbed")
    p.add_argument("--rates", type=float, nargs="+", default=[.05, .1, .2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-size", type=int, default=30)
    p = sub.add_parser("robustness")
    p.add_argument("--run", required=True)
    p.add_argument("--clean", default="data/test.jsonl")
    p.add_argument("--perturbed", default="data/perturbed")
    p.add_argument("--ood", default=None)
    p.add_argument("--output", default="runs/robustness_seed42")
    p.add_argument("--device", default="auto")
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("report", help="Export small measured reports suitable for GitHub")
    p.add_argument("--run", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="reports/textcnn_seed42")
    p = sub.add_parser("smoke", help="Tiny synthetic integration run, never benchmark evidence")
    p.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.output, args.seed, args.ood, args.ood_limit, args.revision)
    elif args.command == "train":
        from .engine import train
        train(args.data, args.run, args.config)
    elif args.command == "evaluate":
        from .engine import evaluate
        evaluate(args.run, args.input, args.output, args.name, args.device)
    elif args.command == "perturb":
        from .perturb import generate
        if args.review_size < 1:
            parser.error("review-size must be positive")
        generate(args.input, args.output, args.rates, args.seed, args.review_size)
    elif args.command == "robustness":
        from .robustness import run_suite
        run_suite(args.run, args.clean, args.perturbed, args.output, args.ood, args.device)
    elif args.command == "predict":
        from .engine import predict
        print(json.dumps(predict(args.run, args.text, args.device), indent=2))
    elif args.command == "report":
        from .report import export_report
        export_report(args.run, args.suite, args.data, args.output)
    else:
        from .smoke import smoke
        smoke(args.output)


if __name__ == "__main__":
    main()
