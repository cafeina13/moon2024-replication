"""Download the replication's input data into this notebook's output.

Other notebooks attach this notebook's output as their input data.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

OUT = Path("/kaggle/working")

SOURCES = {
    # UCI Appliances Energy Prediction (id 374), mirrored by the dataset's author
    "energydata_complete.csv": "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv",
    # The authors' prebuilt feature tables
    "Appliances Energy Prediction.csv": "https://raw.githubusercontent.com/sodayeong/PLOS-ONE_Github/main/dataset/Appliances%20Energy%20Prediction.csv",
    "University Residential Complex.csv": "https://raw.githubusercontent.com/sodayeong/PLOS-ONE_Github/main/dataset/University%20Residential%20Complex.csv",
}
EXPECTED_ROWS = {
    "energydata_complete.csv": 19735,
    "Appliances Energy Prediction.csv": 3289,
    "University Residential Complex.csv": 29232,
}

manifest = {}
for name, url in SOURCES.items():
    body = urllib.request.urlopen(url, timeout=60).read()
    rows = body.count(b"\n") - 1 + (not body.endswith(b"\n"))
    assert rows == EXPECTED_ROWS[name], f"{name}: {rows} rows, expected {EXPECTED_ROWS[name]}"
    (OUT / name).write_bytes(body)
    manifest[name] = {"url": url, "bytes": len(body), "rows": rows, "sha256": hashlib.sha256(body).hexdigest()}
    print(f"{name}: {rows} rows, {len(body):,} bytes")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
