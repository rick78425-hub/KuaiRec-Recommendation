"""Local entry point: no network calls, original inputs are read only."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent

#parser负责把在终端里输入的 `--config`、`--stage`、`--user-id` 这类命令行参数，转换成 Python 代码里可以直接调用的变量，相当于终端和业务代码之间的 “参数翻译官”
#打包成一个 `args` 对象，代码里直接写 `args.user_id`、`args.stage` 就能取值。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=['recommend', 'verify', 'evaluate-val', 'train-v3', 'rebuild-val-state', 'rebuild-test-state', 'export-bundle', 'verify-deepfm', 'evaluate-deepfm', 'audit-test', 'test-end-to-end'])
    parser.add_argument("--stage", choices=["val", "test"], default="val")
    parser.add_argument("--user-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--explain", action="store_true", help="Explain the command without loading data or running computation")
    args = parser.parse_args()
    descriptions = json.loads((ROOT / "configs/commands.json").read_text(encoding="utf-8"))
    if args.explain:
        print(json.dumps(descriptions[args.command], ensure_ascii=False, indent=2))
        return
    scripts = {'rebuild-val-state': 'data/rebuild_val_state.py', 'rebuild-test-state': 'data/rebuild_test_state.py', 'export-bundle': 'artifacts/export_bundle.py', 'verify-deepfm': 'evaluation/deepfm_verification.py', 'evaluate-deepfm': 'evaluation/deepfm_validation.py', 'audit-test': 'evaluation/test.py', 'test-end-to-end': 'test_end_to_end.py'}
    if args.command in scripts:
        if args.config != "configs/local.json":
            parser.error("This migrated script currently uses configs/local.json; a custom --config is unsupported")
        if args.stage != "val" or args.user_id != 0 or args.top_k != 50 or args.batch_size != 32:
            parser.error("This command takes only its name; stage/user/top-k/batch overrides are unsupported")
        import runpy
        print("Running:", descriptions[args.command]["action"], flush=True)
        old_argv = sys.argv
        try:
            sys.argv = [str(ROOT / scripts[args.command])]
            runpy.run_path(sys.argv[0], run_name="__main__")
        finally:
            sys.argv = old_argv
        return

    if args.command == "evaluate-val" and (args.stage != "val" or args.batch_size < 1):
        parser.error("evaluate-val requires --stage val and a positive batch size")
    if args.command == "train-v3" and args.stage != "val":
        parser.error("train-v3 does not accept --stage test")
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.command not in config.get("supported_commands", ["recommend", "verify", "evaluate-val", "train-v3"]):
        parser.error("The selected configuration supports: " + ", ".join(config["supported_commands"]))
    artifact = Path(config["artifact_dir"]).resolve()
    processed = Path(config["processed_dir"]).resolve()
    output_root = (ROOT / config["output_dir"]).resolve()
    if not output_root.is_relative_to(ROOT) or any(output_root.is_relative_to(p) or p.is_relative_to(output_root) for p in [artifact, processed]):
        raise ValueError("Output must be inside this project and separate from original inputs")
    output = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    if args.command == "train-v3":
        from ranking.lightgbm_ranker import train_v3
        passed = train_v3(processed, artifact, output, ROOT)
        print("Output:", output)
        if not passed:
            raise SystemExit(1)
        return
    from ranking.lightgbm_ranker import load_pipeline
    pipe = load_pipeline(artifact, args.stage)
    if args.command == "evaluate-val":
        from evaluation.validation import evaluate_val
        passed = evaluate_val(pipe, processed, artifact, output, args.batch_size)
        print("Output:", output)
        if not passed:
            raise SystemExit(1)
        return
    if args.command == "recommend":
        result = pipe.recommend(args.user_id, args.top_k)
        result.to_csv(output / "recommendations.csv", index=False)
        print("Recommendation saved locally:", output)
    else:
        from evaluation.reference import verify_reference
        checks = verify_reference(pipe, artifact)
        (output / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        if not checks["passed"]:
            raise SystemExit(1)
    print("Output:", output)

if __name__ == "__main__":
    main()
