# Creating the 75th percentile seasonal model only from Kraemer data set
## Step 1 
- Filename: OCSVM_monthly_model_libsvm_Step_1.py
- This file creates the 75th percentile model from Kraemer data set for both Ae. albopictus and Ae. aegypti
---
# Appending GMOD data set with climate, LULC and population density data
## Step 2.1
- Appends climate data to the GMOD data base for Ae. albopictus and Ae. aegypti
---
## Step 2.2
- Appends LULC and population density data to the GMOD data base for Ae. albopictus and Ae. aegypti
---
# Updating the 75th percentile seasonal model with GMOD data (example for one yearly update)
## Step 3
- Filename: GMOD_online_learning_Step_3.ipynb
- Gathers the data from GMOD that are misclassified by the model
## Step 4
- Filename: GMOD_online_learning_Step_4.ipynb
- Update the model by adding the misclassified data
- This will update the model by each year
