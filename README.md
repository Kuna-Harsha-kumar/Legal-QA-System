The current model is using CAUD database.The model can be trained only on a single document.The below are the sample results : 


The below results shows the results when there are labels in the document,so that supervised method is followed

<xgboost.core.Booster object at 0x788777f99a00>

🟢 DETECTED MODE: SUPERVISED CLASSIFICATION
📌 Evaluating using true labels (no train/test split)

📊 SUPERVISED METRICS:
Accuracy :  0.5510
Precision: 0.4123
Recall   : 0.5510
F1 Score : 0.4510

The below results show  the results when there are no labels and model is following unsupervised method(regression)
