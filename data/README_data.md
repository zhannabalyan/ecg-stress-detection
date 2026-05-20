# Data Directory Documentation

This directory contains the primary datasets and supplementary metadata required to execute the stress detection pipeline. 

##  Critical Storage Warning: Action Required
To comply with Moodle's strict **100 MB** submission file size limit, the heavy raw `.pkl` files (totaling 1.34 GB) have been removed from this local folder and are hosted securely on Google Drive. 

Before running any pipeline code, running notebooks, or launching the Streamlit dashboard, you must download the data files to this directory.

### Local Data Setup Instructions:
1. **Download the Data Pack:** Click the link below to grab the compressed data archive:
    [Download capstone_data.zip from Google Drive](https://drive.google.com/file/d/1JHKXh_3visoikeJaQ8X5dYJgHvEVM0mb/view?usp=drive_link)
2. **Extract the Files:** Unzip `capstone_data.zip`. You will see the 15 subject files (`S2.pkl` through `S17.pkl`).
3. **Deploy the Files:** Move all 15 `.pkl` files directly into this `data/` folder (sitting right next to this `README_data.md` file).

---

## 1. Included Datasets and File Structure

### Physiological Data (`.pkl` files)
Once downloaded from the Google Drive link above, the root of this directory will contain the raw physiological data for 15 subjects from the WESAD dataset, stored as Python pickle (`.pkl`) files:
`S2.pkl`, `S3.pkl`, `S4.pkl`, `S5.pkl`, `S6.pkl`, `S7.pkl`, `S8.pkl`, `S9.pkl`, `S10.pkl`, `S11.pkl`, `S13.pkl`, `S14.pkl`, `S15.pkl`, `S16.pkl`, `S17.pkl`

*(Note: Subject S12 is deliberately excluded from this repository, as data was not successfully collected for this individual by the original researchers).*

### Subject Metadata (`metadata/` folder)
Alongside the sensor data, this directory includes a `metadata/` subfolder. This folder contains the demographic information and psychological questionnaire responses (e.g., STAI, PANAS, SAM) completed by each subject during the original study. This provides contextual background for the physiological stress responses observed in the `.pkl` files.

---

## 2. Original Source and Attribution
While the datasets are hosted on Google Drive for submission convenience, the original WESAD (Wearable Stress and Affect Detection) dataset is publicly available for academic research. 

* **Public Repository / Download Link:** [WESAD Dataset via Sciebo](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx)
* **Citation:** Schmidt, P., Reiss, A., Dürichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection*. Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI 2018). https://doi.org/10.1145/3242969.3242985

---

## 3. Variable Descriptions
Each `.pkl` file contains a nested dictionary encompassing various sensor modalities. For the scope of this project, the pipeline isolates and analyzes the following variables:

* **`signal['chest']['ECG']`**: Continuous raw Electrocardiogram (ECG) sensor values, sampled at a high-fidelity rate of **700 Hz**.
* **`label`**: Ground-truth experimental condition labels, synchronized precisely with the sensor data:
  * `0` = Transient / Unlabeled
  * `1` = Baseline (Neutral relaxation state)
  * `2` = Stress (Trier Social Stress Test - TSST)
  * `3` = Amusement (Watching video clips)
  
*Project Scope Note: Only labels `1` (Baseline) and `2` (Stress) are utilized to frame this as an unsupervised anomaly detection task.*

---

## 4. Preprocessing Steps
Before the raw ECG signals are evaluated by the Matrix Profile and HRV pipelines, the data undergoes the following strict preprocessing regimen:

1. **Condition Segmentation:** The continuous 700 Hz ECG arrays are filtered using the `label` array to dynamically slice and extract only the pure Baseline (`1`) and Stress (`2`) segments.
2. **Chronological Data Splitting:** To prevent data leakage, the Baseline condition is chronologically partitioned into three segments: 
   * **Training:** Used exclusively to discover "normal" baseline cardiac motifs.
   * **Validation:** Used to compute distance profiles and tune the optimal anomaly detection threshold.
   * **Testing:** Held out for final evaluation alongside the Stress condition.
3. **Z-Normalization:** To ensure the Matrix Profile algorithm evaluates morphological shape rather than absolute amplitude, all ECG subsequences are strictly Z-normalized (mean = 0, standard deviation = 1) during the `stumpy` motif discovery phase.
4. **HRV Feature Extraction (Secondary Pipeline):** For the hybrid model analysis, an automated peak-detection algorithm isolates the R-peaks within the ECG signal to calculate Heart Rate Variability (HRV) metrics, specifically RMSSD and mean Heart Rate, across discrete time windows.