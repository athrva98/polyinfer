"""Tests for package metadata consistency.

These guard against the version and license metadata drifting between
``src/polyinfer/__init__.py``, ``pyproject.toml``, and the installed
distribution. All three previously disagreed.
"""

import re
import sys
from pathlib import Path

import pytest

import polyinfer as pi

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    """Load pyproject.toml, skipping if unavailable in this environment."""
    if not PYPROJECT.exists():
        pytest.skip("pyproject.toml not available (installed, not a source checkout)")
    if sys.version_info < (3, 11):
        pytest.importorskip("tomli", reason="tomllib requires Python 3.11+")
        import tomli as toml_reader
    else:
        import tomllib as toml_reader

    with open(PYPROJECT, "rb") as f:
        return toml_reader.load(f)


class TestVersion:
    """The package version must have exactly one source of truth."""

    def test_version_is_defined(self):
        assert isinstance(pi.__version__, str)
        assert pi.__version__

    def test_version_is_pep440(self):
        # Simple N.N.N[suffix] check - enough to catch placeholders like "unknown".
        assert re.match(r"^\d+\.\d+\.\d+", pi.__version__), (
            f"__version__ is not a valid version: {pi.__version__!r}"
        )

    def test_pyproject_derives_version_dynamically(self):
        """pyproject must NOT hardcode a second copy of the version."""
        data = _load_pyproject()
        project = data["project"]

        assert "version" not in project, (
            "pyproject.toml hardcodes a version. It must declare "
            'dynamic = ["version"] and let hatchling read '
            "src/polyinfer/__init__.py so there is a single source of truth."
        )
        assert "version" in project.get("dynamic", []), (
            'pyproject.toml must declare dynamic = ["version"]'
        )

    def test_installed_metadata_matches_dunder_version(self):
        """The built/installed distribution must report the same version."""
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as dist_version

        try:
            installed = dist_version("polyinfer")
        except PackageNotFoundError:
            pytest.skip("polyinfer is not installed in this environment")

        assert installed == pi.__version__, (
            f"Installed distribution reports {installed!r} but "
            f"polyinfer.__version__ is {pi.__version__!r}"
        )

    def test_cli_reports_package_version(self, capsys):
        """`polyinfer --version` must not carry its own hardcoded string."""
        from polyinfer.cli import main

        sys_argv = sys.argv
        try:
            sys.argv = ["polyinfer", "--version"]
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        finally:
            sys.argv = sys_argv

        assert pi.__version__ in capsys.readouterr().out


class TestLicense:
    """License metadata must agree with the LICENSE file (Apache-2.0)."""

    def test_license_field_is_apache(self):
        data = _load_pyproject()
        assert data["project"]["license"] == "Apache-2.0"

    def test_license_classifier_matches_license_field(self):
        data = _load_pyproject()
        classifiers = data["project"].get("classifiers", [])
        license_classifiers = [c for c in classifiers if c.startswith("License ::")]

        assert license_classifiers == ["License :: OSI Approved :: Apache Software License"], (
            f"License classifier disagrees with license = 'Apache-2.0': {license_classifiers}"
        )

    def test_license_file_is_apache(self):
        license_file = PYPROJECT.parent / "LICENSE"
        if not license_file.exists():
            pytest.skip("LICENSE not available")
        assert "Apache License" in license_file.read_text(encoding="utf-8")[:200]
