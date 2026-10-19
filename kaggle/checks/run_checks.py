"""Run the repository's test suite on Kaggle against the data from moon2024-get-data."""

import os
import subprocess
import sys

REPO = "https://github.com/cafeina13/moon2024-replication"
SRC = "/kaggle/tmp/moon2024-replication"


def run(*cmd, **kwargs):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


run("git", "clone", "--depth", "1", REPO, SRC, check=True)
run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{SRC}[test]", check=True)
result = run(
    sys.executable, "-m", "pytest", "-v", "--junitxml=/kaggle/working/pytest.xml",
    cwd=SRC, env={**os.environ, "MOON_DATA_DIR": "/kaggle/input"},
)
sys.exit(result.returncode)
