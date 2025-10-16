#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw

# deps (skip if already installed)
uv add -q ucimlrepo pandas pyarrow openml >/dev/null 2>&1 || true

echo "== UCI via ucimlrepo: Car (id=19) & Covertype (id=31) =="
python - <<'PY'
from ucimlrepo import fetch_ucirepo
import pandas as pd, pathlib as p
out = p.Path("raw"); out.mkdir(exist_ok=True)

def dump(did, name):
    ds = fetch_ucirepo(id=did)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(out/f"{name}.csv", index=False)
    print(f"✓ {name}: {df.shape}")
dump(19, "car")
dump(31, "covertype")
PY

echo "== UCI direct ZIP: APS Failure (id=421) =="
# Official download button target from UCI page
# https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
APS_ZIP="https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
curl -fL --retry 5 --retry-delay 3 "$APS_ZIP" -o raw/aps.zip
unzip -o raw/aps.zip -d raw >/dev/null
rm -f raw/aps.zip
# Normalize filenames
[ -f raw/aps_failure_training_set.csv ] && mv raw/aps_failure_training_set.csv raw/aps_training.csv
[ -f raw/aps_failure_test_set.csv ] && mv raw/aps_failure_test_set.csv raw/aps_test.csv
echo "✓ APS: raw/aps_training.csv + raw/aps_test.csv"

echo "== OpenML: Jannis (id=41168), retrying up to 5x =="
python - <<'PY'
import openml, pandas as pd, time, pathlib as p
out = p.Path("raw")
for a in range(1,6):
    try:
        d = openml.datasets.get_dataset(41168)  # Jannis
        X,y,_,_ = d.get_data(target=d.default_target_attribute, dataset_format="dataframe")
        pd.concat([X, y.rename("__target__")], axis=1).to_csv(out/"jannis.csv", index=False)
        print(f"✓ Jannis: {X.shape[0]}x{X.shape[1]} -> raw/jannis.csv"); break
    except Exception as e:
        print(f"Jannis attempt {a}/5: {e}"); time.sleep(2**a)
else:
    print("⚠ Jannis still failing (OpenML availability). You can rerun this script later.")
PY

echo "== Done. Files =="
ls -lh raw
