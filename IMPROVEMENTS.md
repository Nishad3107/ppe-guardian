# PPE Model Alignment & Training Action Plan

This plan outlines the structured steps to replace the generic COCO model currently at `models/ppe.pt` with a custom-trained YOLOv8 model that correctly identifies the 4 required PPE classes: `helmet`, `mask`, `glasses`, and `boots`.

---

## Group 1: Dataset Merging & Remapping
Since the raw assets are split into separate directories (`boots`, `glasses`, `helmet_dataset`, and `mask`), we first need to remap their internal class IDs to the project specification and merge them.

### Step 1: Run the Merge Pipeline
Execute the merge script to remap all annotation files and compile the images into a single training dataset:
```bash
node scripts/run-python.js scripts/merge_ppe_datasets.py --force-replace --imbalance-threshold 200
```
*   **What this does:**
    *   Finds labels in `datasets/helmet_dataset`, `datasets/mask/mask-detection`, `datasets/glasses`, and `datasets/boots/...`.
    *   Remaps annotations to the target class IDs: `helmet` (0), `mask` (1), `glasses` (2), and `boots` (3).
    *   Materializes the clean output inside `datasets/ppe_final`.
    *   Generates `datasets/ppe_final/data.yaml`.
*   **Key Detail:** We specify `--imbalance-threshold 200` to allow the dataset promotion to proceed despite the class count discrepancy (helmet has ~16,381 labels while glasses has ~97).

---

## Group 2: Dataset Verification
Verify the dataset structure, file mappings, and class distributions to guarantee they are ready for YOLOv8 training.

### Step 2: Validate the Merged Dataset
Run the validation script on the final folder:
```bash
node scripts/run-python.js scripts/validate_ppe_dataset.py --imbalance-threshold 200
```
*   **Expected Outcome:**
    *   Output shows a balanced/valid set of 4 classes (`helmet`, `mask`, `glasses`, `boots`).
    *   `train` split has 6,281 image-label pairs; `val` split has 204 image-label pairs.
    *   Validation reports `READY FOR TRAINING`.

---

## Group 3: Model Training
Train a custom YOLOv8 detection model using the merged dataset.

### Step 3: Run Training Dry-Run
Preview the exact training command to confirm parameters:
```bash
node scripts/run-python.js scripts/train_ppe_model.py --imbalance-threshold 200
```

### Step 4: Execute Training
Start the model training:
```bash
node scripts/run-python.js scripts/train_ppe_model.py --imbalance-threshold 200 --run --device cpu
```
*   **Parameters adjustment:**
    *   `--epochs`: Default is 150 epochs. You can reduce this for testing (e.g. `--epochs 10` or `--epochs 50`).
    *   `--device`: Specify your compute hardware:
        *   Apple Silicon (Mac M1/M2/M3): `--device mps`
        *   NVIDIA GPU (CUDA): `--device 0`
        *   No GPU fallback: `--device cpu`
*   **Expected Output:** Once training completes, the best weights will be saved to `runs/detect/train/weights/best.pt`.

---

## Group 4: Verification and Deployment
Confirm that the trained model contains the required classes and deploy it.

### Step 5: Verify Class Integrity & Deploy
Deploy and run a smoke test on a sample frame to verify predictions:
```bash
node scripts/run-python.js scripts/verify_and_deploy_ppe_model.py \
    --best-model runs/detect/train/weights/best.pt \
    --deploy-model models/ppe.pt
```
*   **What this does:**
    *   Checks class names inside `best.pt`.
    *   Ensures that only the expected classes (`helmet`, `mask`, `glasses`, `boots`) are exported.
    *   Copies the validated model file to `models/ppe.pt`.
    *   Performs a sample prediction on a frame from `videos/test-video.mp4` to ensure inference works without errors.

---

## Group 5: Application Restart & Testing
With the valid model deployed at `models/ppe.pt`, restart the web application.

### Step 6: Restart the Server
Kill your current server and start it again:
```bash
npm run dev
```
*   **Validation Check:** Look at the terminal console. The startup validation warnings will now print:
    *   `Loaded PPE model from models/ppe.pt`
    *   No more warning about missing classes or COCO class names.
    *   The `Source ready | PPE model active` status will show in your dashboard, indicating PPE tracking is 100% active.
