# QR-Legitimacy1

A Python-based toolkit for assessing the legitimacy of QR codes.  
It includes scripts for dataset generation, feature extraction, model training, and evaluation.

---

## 🗂 Project Structure

- **features_upi.py** → Extracts features from UPI-related QR code data.  
- **generate_dataset.py** → Generates/organizes datasets of QR codes (legit/malicious).  
- **generate_demo_qrs.py** → Creates demo QR codes for testing.  
- **train_model.py** → Trains a machine learning model to classify QR codes.  
- **test_model.py** → Evaluates the trained model using test data.  
- **test_all_urls.py** → Tests model inference across all URLs in a dataset.  
- **all_results.json** → Stores aggregated evaluation results.  
- **training_metrics.json** → Contains training performance metrics (accuracy, loss, etc.).

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.7+  
- Suggested libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `qrcode`
  - `opencv-python`

Install dependencies:
```bash
pip install numpy pandas scikit-learn qrcode opencv-python
Usage
1. Generate Dataset
python generate_dataset.py

2. Extract Features
python features_upi.py

3. Train Model
python train_model.py


➡ Results saved in training_metrics.json.

4. Test Model
python test_model.py


➡ Results saved in all_results.json.

5. Batch Test URLs
python test_all_urls.py

6. Generate Demo QR Codes
python generate_demo_qrs.py

Results

Training metrics → training_metrics.json

Evaluation results → all_results.json
```bash
pip install numpy pandas scikit-learn qrcode opencv-python
