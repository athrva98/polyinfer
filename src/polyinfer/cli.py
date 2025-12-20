"""Command-line interface for PolyInfer."""

import argparse
import sys


def cmd_info(args):
    """Show system information."""
    import json

    import polyinfer as pi

    info = pi.discovery.system_info()

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("PolyInfer System Information")
        print("=" * 50)
        print(f"\nPlatform: {info['system']['platform']}")
        print(f"Python: {info['system']['python_version'].split()[0]}")
        print(f"Architecture: {info['system']['architecture']}")

        print("\nBackends:")
        for name, backend_info in info["backends"].items():
            status = "OK" if backend_info.get("available") else "NOT AVAILABLE"
            version = backend_info.get("version", "")
            print(f"  {name}: {status} {f'(v{version})' if version else ''}")
            if backend_info.get("available"):
                devices = backend_info.get("devices", [])
                print(f"    Devices: {', '.join(devices)}")

        print("\nAvailable Devices:")
        for device in info["devices"]:
            print(f"  {device['name']}: {', '.join(device['backends'])}")


def cmd_benchmark(args):
    """Benchmark a model."""
    import numpy as np

    import polyinfer as pi

    # Parse input shape
    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    if args.backend:
        # Single backend benchmark
        results = pi.benchmark(
            args.model,
            inputs=np.random.rand(*input_shape).astype(np.float32),
            backend=args.backend,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if results["status"] == "success":
            print(f"Backend: {results['backend']}")
            print(f"Device: {results['device']}")
            print(f"Mean: {results['mean_ms']:.2f} ms")
            print(f"Std: {results['std_ms']:.2f} ms")
            print(f"FPS: {results['fps']:.1f}")
        else:
            print(f"Error: {results['error']}")
    else:
        # Compare all backends
        pi.compare(
            args.model,
            input_shape=input_shape,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            verbose=True,
        )


def cmd_run(args):
    """Run inference on a model."""
    import numpy as np

    import polyinfer as pi

    # Load model
    model = pi.load(args.model, device=args.device, backend=args.backend)
    print(f"Loaded: {model}")
    print(f"Inputs: {model.input_names} {model.input_shapes}")
    print(f"Outputs: {model.output_names}")

    # Generate random input
    if model.input_shapes and all(isinstance(d, int) and d > 0 for d in model.input_shapes[0]):
        shape = model.input_shapes[0]
    else:
        shape = tuple(int(x) for x in args.input_shape.split(","))

    input_data = np.random.rand(*shape).astype(np.float32)
    print(f"\nInput shape: {input_data.shape}")

    # Run inference
    output = model(input_data)
    if isinstance(output, tuple):
        for i, o in enumerate(output):
            print(f"Output {i}: shape={o.shape}, dtype={o.dtype}")
    else:
        print(f"Output: shape={output.shape}, dtype={output.dtype}")


def main():
    parser = argparse.ArgumentParser(
        prog="polyinfer",
        description="PolyInfer - Unified ML inference across multiple backends",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a model")
    bench_parser.add_argument("model", help="Path to ONNX model")
    bench_parser.add_argument("--device", "-d", default="cpu", help="Target device")
    bench_parser.add_argument("--backend", "-b", help="Specific backend to use")
    bench_parser.add_argument(
        "--input-shape", "-s", default="1,3,224,224", help="Input shape (comma-separated)"
    )
    bench_parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup iterations")
    bench_parser.add_argument(
        "--iterations", "-n", type=int, default=100, help="Benchmark iterations"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference")
    run_parser.add_argument("model", help="Path to ONNX model")
    run_parser.add_argument("--device", "-d", default="cpu", help="Target device")
    run_parser.add_argument("--backend", "-b", help="Specific backend to use")
    run_parser.add_argument(
        "--input-shape", "-s", default="1,3,224,224", help="Input shape (comma-separated)"
    )

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
