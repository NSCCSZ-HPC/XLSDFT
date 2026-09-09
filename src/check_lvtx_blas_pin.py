#!/usr/bin/env python3
"""Fail closed unless the shared LVTX header tree is the reviewed snapshot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


_DEFAULT_ROOTS = (
    "/home/ubuntu/sme/gbkernels/home/lvtx_blas",
    "/home/share/shenchao_common/lvtx_blas",
)


def resolve_expected_root() -> Path:
    override = os.environ.get("LVTX_BLAS_ROOT", "").strip()
    if override:
        return Path(override)
    for candidate in _DEFAULT_ROOTS:
        path = Path(candidate)
        if path.is_dir():
            return path
    return Path(_DEFAULT_ROOTS[0])


EXPECTED_ROOT = resolve_expected_root()
EXPECTED_COMMIT = "3a068939753c68d412dc0b7154c21ec3f41cf8da"
EXPECTED_HASHES = {
    "SOURCE_MANIFEST.sha256":
        "25c7d95ebe39130494920730746e6a3f8287f9697f117dd1e208c4fc3b10b362",
    "UPSTREAM_MANIFEST.sha256":
        "0b3b78857dacd13255713a4020aea63de63d778614fd819522a39be044ed1ca3",
    "lvtx_blas.hpp":
        "5972b6ffd4d3823c0a78bf266b403ea439aa1cd92614b6c189d530c9c5235c2c",
}


class PinError(RuntimeError):
    pass


def run_checked(command: list[str], cwd: Path) -> str:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PinError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> None:
    try:
        resolved_root = EXPECTED_ROOT.resolve(strict=True)
    except OSError as error:
        raise PinError(f"cannot resolve {EXPECTED_ROOT}: {error}") from error
    git_root = Path(
        run_checked(["git", "rev-parse", "--show-toplevel"], resolved_root)
    ).resolve(strict=True)
    if git_root != resolved_root:
        raise PinError(f"Git root is {git_root}, expected {resolved_root}")

    commit = run_checked(["git", "rev-parse", "HEAD"], resolved_root)
    if commit != EXPECTED_COMMIT:
        raise PinError(f"commit is {commit}, expected {EXPECTED_COMMIT}")

    status = run_checked(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        resolved_root,
    )
    if status:
        raise PinError(f"Git tree is not clean: {status!r}")

    for relative_name, expected_hash in EXPECTED_HASHES.items():
        path = resolved_root / relative_name
        if not path.is_file():
            raise PinError(f"missing pinned file: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise PinError(
                f"{relative_name} sha256 is {actual_hash}, expected {expected_hash}"
            )

    # SOURCE_MANIFEST covers every delivered dependency file, including the
    # pinned UPSTREAM_MANIFEST file itself.  Do not dereference the latter:
    # its entries are provenance paths outside this self-contained package and
    # are not runtime/build dependencies of XLSDFT.
    run_checked(
        ["sha256sum", "--strict", "-c", "SOURCE_MANIFEST.sha256"],
        resolved_root,
    )

    print(
        "LVTX_BLAS_PIN_OK"
        f" root={resolved_root}"
        f" commit={commit}"
        f" source_manifest_sha256={EXPECTED_HASHES['SOURCE_MANIFEST.sha256']}"
        f" upstream_manifest_sha256={EXPECTED_HASHES['UPSTREAM_MANIFEST.sha256']}"
        f" header_sha256={EXPECTED_HASHES['lvtx_blas.hpp']}"
    )


def main() -> int:
    try:
        verify()
    except (OSError, PinError) as error:
        detail = str(error).replace("\n", "\\n")
        print(f"LVTX_BLAS_PIN_ERROR detail={detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
