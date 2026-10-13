# Train the parking-slot detector on Kaggle

This guide uses the repository's `Parking_dataset_2` and the notebook at
`notebooks/ParkingDetection_Kaggle.ipynb`.

## Important limitation

`Parking_dataset_2` contains parking-space boxes labeled `space-empty` and
`space-occupied`. It does not contain individual car boxes or
correct-parking/wrong-parking labels. This workflow trains **only the slot
occupancy detector**. It does not train a car detector or improve the
correct/wrong-parking classifier. Those decisions still depend on vehicle
detections and camera-specific slot geometry.

## 1. Prepare the code upload

Open PowerShell in the repository root and create the code archive:

```powershell
cd "C:\Users\Yug\Desktop\parking-detection"
python scripts\make_kaggle_code_zip.py
```

This creates `parking-detection-code.zip`. Upload it to Kaggle as a dataset,
for example `parking-detection-code`. The training notebook extracts this
archive automatically.

## 2. Add `Parking_dataset_2` to Kaggle

Create a Kaggle dataset named e.g. `parking-dataset-2`, using the contents of
the local `Parking_dataset_2` directory. Keep the dataset structure intact:

```text
data.yaml
train/
  images/
  labels/
valid/
  images/
  labels/
test/
  images/
  labels/
```

You can upload the directory contents directly or upload a ZIP whose top level
contains `data.yaml` and the three split directories. The notebook can use
either layout and extracts a single ZIP automatically.

## 3. Create and configure the Kaggle notebook

1. In Kaggle, create a new notebook or import
   `notebooks/ParkingDetection_Kaggle.ipynb`.
2. In **Settings**, select **GPU** as the accelerator and enable **Internet**.
   Restart the session if Kaggle asks.
3. Use **Add Input** to attach both Kaggle datasets:
   `parking-detection-code` and `parking-dataset-2`.
4. Run the first cell and check the input folder names it prints.
5. In the first cell, change `CODE_INPUT` and `DATA_INPUT` if your Kaggle
   dataset slugs differ from:

   ```python
   CODE_INPUT = Path('/kaggle/input/parking-detection-code')
   DATA_INPUT = Path('/kaggle/input/parking-dataset-2')
   ```

## 4. Run the notebook from top to bottom

The notebook installs the required packages, checks the GPU and dataset class
mapping, then trains one YOLOv8s slot detector. Defaults are:

| Setting | Value |
|---|---:|
| Training images | 8,691 |
| Validation images | 2,483 |
| Held-out test images | 1,242 |
| Epochs | 50 (early stopping may stop sooner) |
| Image size | 640 |
| Batch | 16 |
| GPU device | 0 |
| Seed | 42 |

The final evaluation uses the test split and reports **slot-detection**
metrics. These metrics are not wrong-parking accuracy. If Kaggle reports a GPU
out-of-memory error, change `BATCH = 16` to `8` or `4`, then rerun the
training cell.

## 5. Download the outputs

After the final notebook cell finishes, use **Save Version → Save & Run All**
so Kaggle preserves the output files. Then download these files from the
notebook's output files:

```text
parking-detection/weights/slot_det_best.pt
parking-detection/runs/slot_det/results.csv
parking-detection/outputs/evaluation/slot_metrics.json
```

Copy `slot_det_best.pt` into this local repository's `weights` directory.
Then run inference on an image or image folder, for example:

```powershell
cd "C:\Users\Yug\Desktop\parking-detection"
python src\inference.py `
  --source "C:\path\to\parking-image.jpg" `
  --slot-weights weights\slot_det_best.pt `
  --out outputs\inference\kaggle-trained
```

The Kaggle run creates no vehicle weights. Keep using the project's existing
vehicle-detection option; do not rename the slot model to `car_det`.

## Troubleshooting

| Problem | What to check |
|---|---|
| CUDA unavailable | Select GPU in notebook Settings and restart the session. |
| `Could not find src/train.py` | Check `CODE_INPUT` and that the code ZIP is attached. |
| Dataset folders not found | Check `DATA_INPUT`; the data YAML and train/valid/test folders must be in the uploaded dataset or ZIP. |
| Class mapping assertion fails | Confirm `data.yaml` lists `space-empty` then `space-occupied`. |
| GPU out of memory | Reduce `BATCH` to `8` or `4`. |
| Pip install fails | Enable Internet in notebook Settings. |
