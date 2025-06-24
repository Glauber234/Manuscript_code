# Manuscript_code
Machine learning models code for the paper: High-content imaging of primary chronic lymphocytic leukemia cells predicts patient cohorts with distinct cellular drug responses.

To predict clinical risk (high vs. low), we analyzed 110 image features. First, we normalized the features and selected the top 40 most important ones using Wasserstein distance (a measure of how different the feature distributions were between high- and low-risk patients).
Next, we used these 40 features to:
1.	Calculate similarities between patients (using mean Wasserstein distance), creating a patient similarity matrix.
2.	Train machine learning models in Python, including:
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- AdaBoost
- Bagging Classifier
- Random Forest
- Gaussian Naive Bayes
- Extra Trees Classifier

We used hierarchical clustering to check how stable the clusters were by removing one patient at a time and reclustering the rest (patient dropout analysis). We then calculated how often each patient was classified as high-risk. Patients with very consistent results (almost always or almost never classified as high-risk) were considered stable.

Enrichment analysis:
Step 1: Group Phenotypes into Bins
- The 7 phenotypes were divided into 10 equal bins (0-10%, 10-20%, ..., 90-100%) based on their values.
- For each cluster, we counted how many patient samples fell into each bin.
Step 2: Simulate Expected Distribution
- To determine what the distribution should look like by random chance, we shuffled the cluster labels 1,000 times and recalculated the bin counts each time.
-	This gave us an average (mean) and variability (standard deviation) for the expected distribution.
Step 3: Compare Observed vs. Expected
-	We compared the actual (observed) distribution of cells in each bin to the simulated (expected) distribution.
-	A Z-score was calculated for each bin to measure how much the observed count deviated from the expected random distribution. This Z-score tells us whether a phenotype is significantly over- or under-represented in a cluster compared to random chance.

