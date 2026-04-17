# Task 9: Multimodal ML - Housing Price Prediction (Images + Tabular)

This task predicts housing prices by combining:

- Image features extracted with a CNN (ResNet18 backbone)
- Structured/tabular housing features from CSV

The model fuses both modalities and performs regression.

## Features

- CNN image encoder (PyTorch + torchvision)
- Tabular preprocessing with `ColumnTransformer`
- Feature fusion (image embedding + tabular embedding)
- Regression evaluation with MAE and RMSE
- Exports predictions, metrics, and training plots

## Dataset (KaggleHub)

This task is configured to use:

- `ted8080/house-prices-and-images-socal`

The script downloads the dataset automatically via `kagglehub`, then finds CSV and image files.

Reference snippet:

```python
import kagglehub

path = kagglehub.dataset_download("ted8080/house-prices-and-images-socal")
print("Path to dataset files:", path)
```

## Run

From project root:

```bash
python Task_09_Final_Phase/task9_multimodal_housing_prediction.py \
  --dataset-source kagglehub \
  --kaggle-dataset ted8080/house-prices-and-images-socal \
  --target-column price \
  --epochs 15
```

If the dataset CSV uses an explicit image column, set it manually:

```bash
python Task_09_Final_Phase/task9_multimodal_housing_prediction.py \
  --dataset-source kagglehub \
  --kaggle-dataset ted8080/house-prices-and-images-socal \
  --target-column price \
  --image-column image_name
```

You can still run with local files (manual paths):

```bash
python Task_09_Final_Phase/task9_multimodal_housing_prediction.py \
  --dataset-source local \
  --input-csv Housing.csv \
  --image-dir Task_09_Final_Phase/images \
  --target-column price \
  --image-column image_name
```

## Outputs

Saved in `Task_09_Final_Phase/outputs` by default:

- `multimodal_metrics.json` (contains MAE and RMSE)
- `actual_vs_predicted_multimodal.csv`
- `training_loss_curve.png`
- `actual_vs_predicted_plot.png`
- `multimodal_regressor.pt`

## Notes

- The script requires at least 40 rows that successfully map to images.
- If the script cannot find images, verify naming conventions and selected mapping column.
- Add `--pretrained-cnn` to start from pretrained ResNet18 weights.

