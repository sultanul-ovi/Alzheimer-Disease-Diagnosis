# Alzheimer Disease MRI Classification

A comprehensive deep learning pipeline for classifying Alzheimer's disease severity from MRI images using the OASIS dataset. Supports multiple state-of-the-art CNN backbones, advanced visualizations, and robust evaluation.

## 🚀 Features
- **Multi-model training:** VGG, ResNet, DenseNet, MobileNetV3, ShuffleNetV2
- **Stratified data split:** 70% train, 15% val, 15% test
- **Class imbalance handling:** Weighted loss
- **Rich visualizations:** Training curves, confusion matrix, ROC, class report
- **Automatic results saving and comparison**
- **Jupyter Notebook workflow**

## 🧠 Dataset
- **Source:** OASIS MRI slices (Kaggle)
- **Classes:**
  - Non Demented
  - Very mild Dementia
  - Mild Dementia
  - Moderate Dementia
- **Images:** ~86,000 JPEGs, organized by class folders

## ⚙️ Setup
1. **Python 3.8+ recommended**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the OASIS dataset from Kaggle (see notebook for path details)
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## 📝 How to Use
1. **Open the notebook** (`notebook.ipynb`)
2. **Run all cells** to verify data loading and environment
3. **To train all models:**
   - Find the cell with the model training loop (see code block below)
   - Adjust `num_epochs` if needed
   - Run the cell to train all models sequentially

```python
all_models = [
    'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet201', 'mobilenetv3_large', 'shufflenet_v2_x1_0'
]
for model_name in all_models:
    train_model(model_name=model_name, data_loaders=data_loaders, ...)
```

4. **Results and models** are saved in the `results/` and `models/` folders.
5. **Visualize and compare** results using the provided analysis cells.

## 🗂️ Output
- **Trained models:** `models/`
- **Metrics/results:** `results/`
- **Plots:** Displayed inline in the notebook

## 🛠️ Troubleshooting
- **Training is slow?**
  - Use a GPU if available (set up CUDA)
  - Reduce `num_epochs` for quick tests
  - Start with lightweight models (MobileNet, ShuffleNet)
- **Out of memory?**
  - Lower batch size
  - Use smaller image size in transforms
- **Dataset not found?**
  - Check the dataset path in the notebook and adjust as needed

## 🙏 Contributor
- Sajib Hossain
