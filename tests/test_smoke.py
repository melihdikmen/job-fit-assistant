"""Smoke test: the package is installed and importable."""

import job_fit


def test_package_is_importable() -> None:
    """The job_fit package imports and exposes its module docstring."""
    assert job_fit.__doc__
