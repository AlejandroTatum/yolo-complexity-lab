# Deployment guide

`app.py` is the canonical entry point for both local and hosted runs. The same
benchmark, model catalog, session state, and CSV export path are used in both
environments; only unsupported capabilities are hidden.

## Prerequisites

- Python 3.11 or newer.
- A clone of this repository and an isolated environment.
- At least 2 GB RAM for the full model comparison; 3 GB is safer on hosted
  runners.
- Network access on the first run so downloadable model weights can be fetched.

Install dependencies from the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The application expects `app.py`, `.streamlit/config.toml`, and
`assets/demo_person_dog_fruit.jpg`. The optional root-level `best.pt` file
enables the custom-weights route; it is not downloaded automatically.

## Local launch

Explicitly select the local capability contract before starting Streamlit:

```bash
YOLOLAB_ENV=local .venv/bin/python -m streamlit run app.py
```

Known model weights are downloaded by the loader into
`~/.cache/yolo-complexity-lab/weights` on first use. Keep the cache between
runs when possible. A local run exposes the webcam, live streaming, CPU/auto,
and CUDA choices when CUDA is available. The demo and image-upload sources
remain available regardless of the camera.

## Hosted launch

Create the Streamlit app from the repository and set the **Main file** to
`app.py`:

1. Open Streamlit Community Cloud and choose **New app**.
2. Select `AlejandroTatum/yolo-complexity-lab`, the target branch, and
   `app.py`.
3. Deploy, then wait for the first-run model downloads to finish.

Do not set `YOLOLAB_ENV=local` on a hosted runner. The hosted environment is
restricted by default, or can be made explicit with `YOLOLAB_ENV=cloud`.

## Capability matrix

| Capability | Local (`YOLOLAB_ENV=local`) | Cloud (`YOLOLAB_ENV=cloud`) |
| --- | --- | --- |
| Demo and image upload | Available | Available |
| Webcam and live streaming | Available when the camera opens | Hidden |
| Device choices | `auto`, `cpu`, and `cuda:0` when CUDA is detected | `cpu` only |
| Custom `best.pt` weights | Available when `best.pt` exists at the repository root | Available only when the file is shipped |
| Standard model downloads | Downloaded on first use | Downloaded on first use |

Unknown environments use the restricted/cloud-safe contract. This prevents
public deployments from rendering controls that cannot work there.

## Verification before release

From the repository root, run the static and server smoke checks plus the full
test suite:

```bash
.venv/bin/python scripts/smoke_check.py
.venv/bin/python scripts/smoke_check.py --server
.venv/bin/pytest -q
```

The smoke check verifies resources, the local/cloud capability matrix, model
catalog contracts, Streamlit startup, `/healthz`, and the primary page. The
frozen benchmark regression keeps exact text/integer fields and the documented
floating-point tolerances intact.

## Recovery

- **Missing dependency:** activate the intended environment and rerun
  `.venv/bin/python -m pip install -r requirements.txt`.
- **Missing model or slow cold start:** allow the first download to finish;
  check the Streamlit logs and available memory before retrying.
- **Camera failure:** grant camera access, connect a local camera, verify the
  selected index, or switch to Demo/Image upload. Hosted runners cannot use a
  browser camera through this server-side path.
- **Corrupt cached weights:** stop the app, remove
  `~/.cache/yolo-complexity-lab/weights`, and rerun the verification command.
- **Smoke failure:** fix the reported prerequisite before sharing the public
  URL; do not treat a failed smoke check as a ready deployment.

## Repointing and rollback

To repoint an existing hosted app, open its settings and change **Main file**
to `app.py`, save, and redeploy. Confirm the server smoke check and a demo
benchmark after the new revision starts.

To roll back, redeploy the last known-good commit or revert the deployment
change in version control, then rerun the same smoke and benchmark gates. No
database or model-data migration is required.
