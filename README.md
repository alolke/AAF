# AAF
**Paper**: "An Adaptive Augmentation Fusion-Based Data Augmentation Method for Abstract Dialogue Summarization."  
This study used two publicly available datasets for the experiments: **DialogSum** and **SAMSum**. These datasets are available for academic research and can be obtained from the relevant project pages or public repositories.

---

### **Project Structure**

This repository is organized as follows:

- **AAF_1_fenju**  
  _Data Processing_: Splitting dialogue texts into the format "Speaker: Dialogue" to facilitate information extraction and generation.

- **AAF_2_1_SR**  
  _Data Augmentation_: Synonym Replacement (SR).

- **AAF_2_2_action_extraction**  
  **AAF_2_2_topic_annotation**  
  **AAF_2_2_ATGEN**  
  _Data Augmentation_: Sequential processes for Action Extraction, Topic Segmentation, and Topic-Action-Based Dialogue Generation Enhancement.

- **AAF_Trend_split**  
  _Fusion Process_: Dataset splitting mechanism for good and bad data during the fusion process.

- **AAF_train_bart.py**  
  _Model Training_: Adaptive fusion augmentation on the BART model.

---


