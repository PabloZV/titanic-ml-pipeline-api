## Pipeline for training Classifier

### Data
- Avoiding redundant encoding: no more dummy variables are needed, since the remaining attributes represent amounts, not classes
- Preventing multicollinearity: one dummy variable from the ones related to Sex, Embarked and Pclass should be dropped to avoid multicollinearity
- Train-serve consistency: source dataset should be transformed to have the same attributes as the expected user input, so the same preprocessing logic during training can be applied to the inference step without repeating code.

### Training
- Ensuring reproducibility: all models should have declared the random seed for reproducibility
- Handling class imbalance: since the data is quite unbalanced regarding 'Sex' and 'Pclass' attributes, the split is made stratified
- Model improvement strategy: To improve the models, there were two approaches, one changing its parameters and the other changing the model for another of the same family. For changing parameters a grid search was used for the LinearSVM and RandomForest. Alternative models from the same family (regressions and trees) were proposed too, for LinearSVM its proposed alternative is a LogisticRegression, and for RandomForest its proposed alternative is XGBoost.
- Fair comparison methodology: the provided classifiers were subject to stratification, and scaling, and f1 was measured instead of accuracy, all this in order to better evaluate it against the improved versions.
- Systematic model selection: the training pipeline, gridsearch kfold stratification and metric comparison is done in /notebooks/2_comparing_models.py . In that notebook it is observed that the best model is an XGBoost, so that's the model that will be used for inference.

### Evaluation
- Metric choice for imbalanced data: it doesn't make sense to use accuracy in unbalanced data like here, a model might learn to just look at the majority class. F1 is preferred in this kind of situations.
- Robust performance estimation: evaluation carried out with cross validation
- Offline model comparison: the comparison between models is done offline in the notebook, then as conclusion the XGBoost model was selected to use in the inference API.
- Interpretability enhancement: the feature importance of dummy variables are summed to match the user input structure, this gives more readability to the importance. It is also expressed into percentages for the same reason. Its saved in /models/feature_importance.json.

### Deployment
- Consistent preprocessing: The same `TitanicInputTransformer` used in training will be used in inference, avoiding train/serve skew
- Model serialization: Models are saved as pickle files for easy loading in production environments
- Feature order consistency: The transformer outputs features in a fixed order that matches expected API input
- Multiple model artifacts: Both XGBoost (best performer) and RandomForest (faster inference) are saved to allow deployment flexibility based on latency vs accuracy trade-offs

### Metric Tracking
- Reproducibility: All hyperparameters, CV scores, and test metrics are saved to recreate results and compare model versions over time
- Model selection justification: Detailed comparison metrics provide evidence for choosing XGBoost over alternatives
- Feature importance transparency: Aggregating dummy variable importances back to original features makes the model more interpretable for business stakeholders
- Performance baseline: Test set evaluation provides unbiased performance estimates for production expectations
- Debugging capability: Storing full CV results allows investigation of model stability and variance across folds

### Monitoring
- Resource profiling during training: CPU and RAM monitoring helps optimize pipeline efficiency and predict infrastructure needs for larger datasets or model retraining
- Structured logging approach: JSON-configurable logging allows different verbosity levels for development vs production, with file rotation to prevent disk space issues
- Exception handling and logging: Comprehensive error capture ensures training failures are traceable and debuggable
- Reproducibility tracking: Fixed seeds, environment specifications, and detailed logging create an audit trail for model development
- Pipeline stage profiling: Measuring each stage separately (data loading, preprocessing, training, evaluation) helps identify bottlenecks and optimize the most expensive operations

### DVC
- Public access Blob Storage: A public accessable GCS Bucket will be used to store DVC Data to facilitate the reproducibililty.


