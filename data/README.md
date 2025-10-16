We selected four tabular classification datasets to stress different parts of the unified preprocessing + HyperParameterOptimization pipeline:
- Jannis (multiclass, mostly numerical): mid/large, good for throughput and optimization behavior at scale.
- Car Evaluation (multiclass, mostly categorical): tiny but pure-categorical, ideal to compare encoding strategies and calibration.
- APSFailure at Scania Trucks (binary, messy): substantial missingness and imbalance to showcase cleaning, imputation, and robust metrics.
- Covertype (multiclass, mixed): medium/large with both numeric and categorical signals, a strong real-world baseline.

This mix gives varied sizes (from small to large) under a ~3 GB cap, with 3 multiclass tasks and exactly 1 binary task, so hyperparameters and learning algorithms can be compared across genuinely different regimes.