#!/usr/bin/env python3
"""Test PolyInfer installation in a fresh virtual environment.

This script:
1. Creates a fresh virtual environment
2. Installs polyinfer with various optional dependencies
3. Verifies imports and basic functionality
4. Runs the test suite
5. Cleans up the virtual environment

Works on both Windows and Linux.

Usage:
    python scripts/test_installation.py                    # Test basic install
    python scripts/test_installation.py --extras cpu       # Test with [cpu]
    python scripts/test_installation.py --extras all       # Test with [all]
    python scripts/test_installation.py --extras examples  # Test with [examples]
    python scripts/test_installation.py --keep-venv        # Don't delete venv after
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_venv_paths(venv_dir: Path) -> dict:
    """Get platform-specific paths for virtual environment."""
    if platform.system() == "Windows":
        return {
            "python": venv_dir / "Scripts" / "python.exe",
            "pip": venv_dir / "Scripts" / "pip.exe",
            "activate": venv_dir / "Scripts" / "activate.bat",
        }
    else:
        return {
            "python": venv_dir / "bin" / "python",
            "pip": venv_dir / "bin" / "pip",
            "activate": venv_dir / "bin" / "activate",
        }


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print output."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n")[:20]:  # Limit output
            print(f"    {line}")
        if result.stdout.count("\n") > 20:
            print(f"    ... ({result.stdout.count(chr(10)) - 20} more lines)")
    if result.stderr and result.returncode != 0:
        for line in result.stderr.strip().split("\n")[:10]:
            print(f"    [stderr] {line}")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def create_venv(venv_dir: Path) -> dict:
    """Create a fresh virtual environment."""
    print(f"\n{'='*60}")
    print(f"Creating virtual environment: {venv_dir}")
    print(f"{'='*60}")

    if venv_dir.exists():
        print(f"  Removing existing venv...")
        shutil.rmtree(venv_dir)

    run_command([sys.executable, "-m", "venv", str(venv_dir)])

    paths = get_venv_paths(venv_dir)

    # Verify python exists
    if not paths["python"].exists():
        raise RuntimeError(f"Python not found at {paths['python']}")

    # Upgrade pip
    print("\n  Upgrading pip...")
    run_command([str(paths["python"]), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    return paths


def install_package(paths: dict, project_dir: Path, extras: str | None = None) -> None:
    """Install polyinfer package."""
    print(f"\n{'='*60}")
    print(f"Installing polyinfer" + (f"[{extras}]" if extras else ""))
    print(f"{'='*60}")

    if extras:
        install_spec = f"{project_dir}[{extras}]"
    else:
        install_spec = str(project_dir)

    run_command([
        str(paths["python"]), "-m", "pip", "install", "-e", install_spec
    ])


def verify_imports(paths: dict, extras: str | None = None) -> bool:
    """Verify that imports work correctly."""
    print(f"\n{'='*60}")
    print("Verifying imports...")
    print(f"{'='*60}")

    # Basic imports that should always work
    basic_imports = """
import polyinfer as pi
print(f"PolyInfer version: {pi.__version__}")
print(f"Available backends: {pi.list_backends()}")
print(f"Available devices: {[d.name for d in pi.list_devices()]}")

# Test core functions exist
assert hasattr(pi, 'load')
assert hasattr(pi, 'list_backends')
assert hasattr(pi, 'list_devices')
assert hasattr(pi, 'export_mlir')
assert hasattr(pi, 'benchmark')
assert hasattr(pi, 'compare')
print("Core API: OK")
"""

    result = run_command(
        [str(paths["python"]), "-c", basic_imports],
        check=False
    )

    if result.returncode != 0:
        print("  FAILED: Basic imports")
        return False

    print("  Basic imports: OK")

    # Check extras-specific imports
    if extras in ["cpu", "all", "nvidia", "intel", "amd"]:
        extras_check = """
import polyinfer as pi
backends = pi.list_backends()
print(f"Backends available: {backends}")

# Should have at least onnxruntime
assert 'onnxruntime' in backends, f"onnxruntime not in {backends}"
print("Backend check: OK")
"""
        result = run_command(
            [str(paths["python"]), "-c", extras_check],
            check=False
        )
        if result.returncode != 0:
            print("  FAILED: Backend check")
            return False
        print("  Backend check: OK")

    return True


def run_tests(paths: dict, project_dir: Path, quick: bool = True) -> bool:
    """Run the test suite."""
    print(f"\n{'='*60}")
    print("Running tests...")
    print(f"{'='*60}")

    # Install pytest
    run_command([
        str(paths["python"]), "-m", "pip", "install", "pytest", "-q"
    ])

    # Run tests
    test_args = [
        str(paths["python"]), "-m", "pytest",
        str(project_dir / "tests"),
        "-v", "--tb=short",
    ]

    if quick:
        # Only run fast tests
        test_args.extend(["-x", "--ignore=tests/test_benchmark.py"])

    result = run_command(test_args, cwd=project_dir, check=False)

    if result.returncode != 0:
        print(f"  Tests FAILED (exit code {result.returncode})")
        return False

    print("  Tests: PASSED")
    return True


def check_installed_packages(paths: dict) -> None:
    """List installed packages for debugging."""
    print(f"\n{'='*60}")
    print("Installed packages:")
    print(f"{'='*60}")

    run_command([
        str(paths["python"]), "-m", "pip", "list", "--format=columns"
    ])


def cleanup_venv(venv_dir: Path) -> None:
    """Remove the virtual environment."""
    print(f"\n{'='*60}")
    print(f"Cleaning up: {venv_dir}")
    print(f"{'='*60}")

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        print("  Removed virtual environment")
    else:
        print("  Virtual environment already removed")


def main():
    parser = argparse.ArgumentParser(description="Test PolyInfer installation")
    parser.add_argument(
        "--extras",
        choices=["cpu", "nvidia", "amd", "intel", "vulkan", "all", "examples", "dev"],
        help="Optional extras to install (e.g., 'cpu', 'all')",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=None,
        help="Directory for virtual environment (default: temp directory)",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Don't delete the virtual environment after testing",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running the test suite",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Run quick tests only (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite",
    )

    args = parser.parse_args()

    # Find project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    if not (project_dir / "pyproject.toml").exists():
        print(f"Error: pyproject.toml not found in {project_dir}")
        sys.exit(1)

    # Determine venv directory
    if args.venv_dir:
        venv_dir = args.venv_dir
    else:
        venv_dir = project_dir / "test_venv"

    print(f"{'#'*60}")
    print(f"# PolyInfer Installation Test")
    print(f"# Platform: {platform.system()} {platform.machine()}")
    print(f"# Python: {sys.version.split()[0]}")
    print(f"# Project: {project_dir}")
    print(f"# Venv: {venv_dir}")
    print(f"# Extras: {args.extras or 'none'}")
    print(f"{'#'*60}")

    success = True

    try:
        # Create virtual environment
        paths = create_venv(venv_dir)

        # Install package
        install_package(paths, project_dir, args.extras)

        # Show installed packages
        check_installed_packages(paths)

        # Verify imports
        if not verify_imports(paths, args.extras):
            success = False

        # Run tests
        if not args.skip_tests and success:
            quick = args.quick and not args.full
            if not run_tests(paths, project_dir, quick=quick):
                success = False

    except Exception as e:
        print(f"\nError: {e}")
        success = False

    finally:
        if not args.keep_venv:
            cleanup_venv(venv_dir)
        else:
            print(f"\nVirtual environment kept at: {venv_dir}")
            print(f"Activate with:")
            if platform.system() == "Windows":
                print(f"  {venv_dir}\\Scripts\\activate")
            else:
                print(f"  source {venv_dir}/bin/activate")

    # Summary
    print(f"\n{'='*60}")
    if success:
        print("INSTALLATION TEST: PASSED")
    else:
        print("INSTALLATION TEST: FAILED")
    print(f"{'='*60}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
