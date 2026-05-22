# 🚀 Machine Learning Notes

Welcome to the **Machine Learning Notes** repository! This is a comprehensive, structured curriculum covering everything from foundational ML concepts, preprocessing, and exploratory data analysis (EDA), to supervised regression, classification, advanced ensemble methods, and unsupervised clustering.

Each guide is designed for clarity, completeness, and practical execution.

---

## 🏃 Quick Start

```bash
# Clone the repository
git clone https://github.com/RiazML/machine-learning-notes.git
cd machine-learning-notes

# Set up the environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn xgboost optuna dtreeviz

# Run syntax and compilation audits
python tests/test_guide.py
```

Each topic follows a **3-step learning structure**:

```
📖 Theory & Math   -> Clean explanations with LaTeX-rendered formulations
📊 Flowcharts      -> Interactive, parse-verified Mermaid diagrams
🐍 Implementations -> Clean, left-aligned, offline-runnable Python code
```

---

## 🗺️ Learning Roadmap & Syllabus

Here is a structured index of the notes. Click on any topic to view the detailed study guide.

### Foundations & Setup

| # | Topic |
| :--- | :--- |
| 001 | [What is Machine Learning?](docs/001_what_is_machine_learning.md) |
| 002 | [AI vs. ML vs. DL](docs/002_ai_vs_ml_vs_dl_for.md) |
| 003 | [Types of Machine Learning](docs/003_types_of_machine_learning_for_beginners.md) |
| 004 | [Batch Machine Learning (Offline Learning)](docs/004_batch_machine_learning.md) |
| 005 | [Online Machine Learning (Incremental Learning)](docs/005_online_machine_learning.md) |
| 006 | [Instance-based vs. Model-based Learning](docs/006_instance-based_vs_model-based_learning.md) |
| 007 | [Challenges in Machine Learning](docs/007_challenges_in_machine_learning.md) |
| 008 | [Real-World Applications of Machine Learning](docs/008_application_of_machine_learning.md) |
| 009 | [Machine Learning Development Life Cycle (MLDLC)](docs/009_machine_learning_development_life_cycle.md) |
| 010 | [Data Roles - Data Engineer vs. Data Analyst vs. Data Scientist vs. ML Engineer](docs/010_data_engineer_vs_data_analyst_vs.md) |
| 011 | [What are Tensors?](docs/011_what_are_tensors.md) |
| 012 | [Installing Anaconda & Development Environments](docs/012_installing_anaconda_&_colab_setup.md) |
| 013 | [End-to-End Machine Learning Toy Project](docs/013_end-to-end_toy_project.md) |
| 014 | [Framing Machine Learning Problems](docs/014_framing_ml_problems.md) |

### Data Preprocessing & EDA

| # | Topic |
| :--- | :--- |
| 015 | [Working with CSV Files in Pandas](docs/015_working_with_csv_files_in_pandas.md) |
| 016 | [Working with JSON & SQL in Pandas](docs/016_working_with_json_sql.md) |
| 017 | [Fetching Data From an API](docs/017_fetching_data_from_an_api.md) |
| 018 | [Fetching Data Using Web Scraping](docs/018_fetching_data_using_web_scraping.md) |
| 019 | [Understanding Your Data](docs/019_understanding_your_data.md) |
| 020 | [Exploratory Data Analysis (EDA) - Univariate Analysis](docs/020_eda_using_univariate_analysis.md) |
| 021 | [EDA using Bivariate & Multivariate Analysis](docs/021_eda_using_bivariate_and_multivariate_analysis.md) |
| 022 | [Automated EDA with Pandas Profiling (`ydata-profiling`)](docs/022_pandas_profiling.md) |
| 023 | [What is Feature Engineering?](docs/023_what_is_feature_engineering.md) |
| 024 | [Feature Scaling - Standardization](docs/024_feature_scaling.md) |
| 025 | [Feature Scaling - Normalization](docs/025_feature_scaling.md) |
| 026 | [Encoding Categorical Data (Ordinal Encoding & Label Encoding)](docs/026_encoding_categorical_data.md) |
| 027 | [One-Hot Encoding (OHE)](docs/027_one_hot_encoding.md) |
| 028 | [ColumnTransformer in Machine Learning](docs/028_column_transformer_in_machine_learning.md) |
| 029 | [Machine Learning Pipelines A–Z](docs/029_machine_learning_pipelines_a-z.md) |
| 030 | [Function Transformer & Mathematical Transforms](docs/030_function_transformer.md) |
| 031 | [Power Transformer: Box-Cox & Yeo-Johnson Transformations](docs/031_power_transformer.md) |
| 032 | [Discretization: Binning & Binarization](docs/032_binning_and_binarization.md) |
| 033 | [Handling Mixed Variables](docs/033_handling_mixed_variables.md) |
| 034 | [Handling Date & Time Variables](docs/034_handling_date_and_time_variables.md) |
| 035 | [Handling Missing Data Part 1: Complete Case Analysis (CCA)](docs/035_handling_missing_data.md) |
| 036 | [Handling Missing Data Part 2: Numerical Mean/Median/Arbitrary Imputation](docs/036_handling_missing_data.md) |
| 037 | [Handling Missing Data Part 3: Categorical Imputation](docs/037_handling_missing_categorical_data.md) |
| 038 | [Missing Indicator & Random Sample Imputation](docs/038_missing_indicator.md) |
| 039 | [K-Nearest Neighbors Imputer (KNNImputer)](docs/039_knn_imputer.md) |
| 040 | [Multivariate Imputation by Chained Equations (MICE)](docs/040_multivariate_imputation_by_chained_equations_for.md) |
| 041 | [What are Outliers?](docs/041_what_are_outliers.md) |
| 042 | [Outlier Detection & Removal Using Z-Score](docs/042_outlier_detection_and_removal_using_z-score.md) |
| 043 | [Outlier Detection & Removal Using the Interquartile Range (IQR) Method](docs/043_outlier_detection_and_removal_using_the.md) |
| 044 | [Outlier Detection Using the Percentile Method](docs/044_outlier_detection_using_the_percentile_method.md) |
| 045 | [Feature Construction & Feature Splitting](docs/045_feature_construction.md) |
| 046 | [The Curse of Dimensionality](docs/046_curse_of_dimensionality.md) |
| 047 | [Principal Component Analysis (PCA): Geometric Intuition](docs/047_principle_component_analysis_pca.md) |
| 048 | [Principal Component Analysis (PCA): Mathematical Derivation](docs/048_principle_component_analysis_pca.md) |
| 049 | [Practical PCA in Scikit-Learn](docs/049_principle_component_analysis_pca.md) |

### Supervised Learning: Regression

| # | Topic |
| :--- | :--- |
| 050 | [Simple Linear Regression (OLS closed-form)](docs/050_simple_linear_regression.md) |
| 051 | [Simple Linear Regression: Geometric Interpretation & Custom Implementation](docs/051_simple_linear_regression.md) |
| 052 | [Regression Evaluation Metrics: Mathematical Derivations & Code Demo](docs/052_regression_metrics.md) |
| 053 | [Multiple Linear Regression: Geometric Intuition & Scikit-Learn Implementation](docs/053_multiple_linear_regression.md) |
| 054 | [Multiple Linear Regression: Mathematical Derivation of the Normal Equation](docs/054_multiple_linear_regression.md) |
| 055 | [Multiple Linear Regression: Coding the Normal Equation from Scratch](docs/055_multiple_linear_regression.md) |
| 056 | [Assumptions of Linear Regression & Diagnostic Tests in Python](docs/056_what_are_the_main_assumptions_of.md) |
| 057 | [Gradient Descent Optimization from Scratch](docs/057_gradient_descent_from_scratch.md) |
| 058 | [Vectorized Batch Gradient Descent for Multiple Linear Regression](docs/058_batch_gradient_descent_with_code_demo.md) |
| 059 | [Stochastic Gradient Descent (SGD)](docs/059_stochastic_gradient_descent.md) |
| 060 | [Mini-Batch Gradient Descent](docs/060_mini-batch_gradient_descent.md) |
| 061 | [Polynomial Regression: Mathematical Formulation & Pipeline Implementation](docs/061_polynomial_regression.md) |
| 062 | [The Bias-Variance Trade-off: Mathematical Decomposition & Simulation](docs/062_bias_variance_trade-off.md) |
| 063 | [Ridge Regression (L2 Regularization): Intuition & Coefficient Shrinkage](docs/063_ridge_regression_part_1.md) |
| 064 | [Ridge Regression: Closed-form Mathematical Matrix Derivation](docs/064_ridge_regression_part_2.md) |
| 065 | [Ridge Regression: Geometric Interpretation & Lagrange Multipliers](docs/065_ridge_regression_part_3.md) |
| 066 | [5 Key Points on Regularization: Theory & Practical Implementation](docs/066_5_key_points.md) |
| 067 | [Lasso Regression (L1 Regularization): Cost Function & Feature Selection](docs/067_lasso_regression.md) |
| 068 | [Why Lasso Regression Creates Sparsity: Geometry & Coordinate Descent](docs/068_why_lasso_regression_creates_sparsity.md) |
| 069 | [Elastic Net Regression: Combining L1 and L2 Regularizations](docs/069_elasticnet_regression.md) |

### Supervised Learning: Classification

| # | Topic |
| :--- | :--- |
| 070 | [Introduction to Logistic Regression: Odds, Log-Odds, & The Sigmoid Function](docs/070_logistic_regression_part_1.md) |
| 071 | [The Perceptron Trick: Binary Classification from a Geometric Perspective](docs/071_logistic_regression_part_2.md) |
| 072 | [Logistic Regression Part 3: Deep Dive into Probability, Odds, Log-Odds, and Sigmoid Symmetry](docs/072_logistic_regression_part_3.md) |
| 073 | [Logistic Regression Part 4: Why MSE Fails, Likelihood, & Binary Cross-Entropy Loss](docs/073_logistic_regression_part_4.md) |
| 074 | [Calculus of the Sigmoid Function: Derivation & Numerical Verification](docs/074_derivative_of_sigmoid_function.md) |
| 075 | [Logistic Regression Part 5: Gradient Derivation & Gradient Descent from Scratch](docs/075_logistic_regression_part_5.md) |
| 076 | [Model Evaluation: Accuracy & The Confusion Matrix](docs/076_accuracy_and_confusion_matrix.md) |
| 077 | [Class Imbalance Metrics: Precision, Recall, & F1-Score](docs/077_precision_recall_and_f1_score.md) |
| 078 | [Model Evaluation: The ROC Curve & AUC](docs/078_roc_curve_in_machine_learning.md) |
| 079 | [Softmax Regression (Multinomial Logistic Regression)](docs/079_softmax_regression.md) |
| 080 | [Non-Linear Boundaries: Polynomial Logistic Regression](docs/080_polynomial_features_in_logistic_regression.md) |
| 081 | [Logistic Regression Hyperparameters: Regularization & Solvers](docs/081_logistic_regression_hyperparameters.md) |
| 082 | [Naive Bayes Classifier Part 1: Conditional Probability](docs/082_naive_bayes_classifier.md) |
| 083 | [Naive Bayes Classifier Part 2: Independent Events](docs/083_naive_bayes_classifier.md) |
| 084 | [Naive Bayes Classifier Part 3: Mutually Exclusive Events](docs/084_naive_bayes_classifier.md) |
| 085 | [Naive Bayes Classifier Part 4: Derivation of Bayes' Theorem](docs/085_naive_bayes_classifier.md) |
| 086 | [Naive Bayes Classifier Part 5: The Law of Total Probability](docs/086_naive_bayes_classifier.md) |
| 087 | [Naive Bayes Classifier Part 6: The "Naive" Assumption](docs/087_naive_bayes_classifier.md) |
| 088 | [Naive Bayes Classifier Part 7: Log Likelihood & Laplace Smoothing](docs/088_naive_bayes_classifier.md) |
| 089 | [Naive Bayes Classifier Part 8: Bag-of-Words Text Classification](docs/089_naive_bayes_classifier.md) |
| 090 | [Naive Bayes Classifier Part 9: Gaussian Naive Bayes](docs/090_naive_bayes_part_9.md) |
| 091 | [K-Nearest Neighbors (KNN) Classifier](docs/091_what_is_k_nearest_neighbors.md) |
| 092 | [Support Vector Machines (Geometric Intuition)](docs/092_support_vector_machines.md) |
| 093 | [Mathematics of SVM - Primal Formulation](docs/093_mathematics_of_svm.md) |
| 094 | [Mathematics of Support Vector Machine: Dual Formulation & KKT Conditions](docs/094_mathematics_of_support_vector_machine.md) |
| 095 | [Mathematics of SVM: Non-linear Mapping & The Kernel Trick](docs/095_kernel_trick_in_svm.md) |
| 096 | [SVM Kernel Functions & Hyperparameter Tuning](docs/096_kernel_trick_in_svm.md) |

### Tree-Based & Ensemble Methods

| # | Topic |
| :--- | :--- |
| 097 | [Decision Trees: Geometric Intuition & Space Partitioning](docs/097_decision_trees_geometric_intuition.md) |
| 098 | [Decision Trees: Split Metrics & Hyperparameter Tuning](docs/098_decision_trees.md) |
| 099 | [Regression Trees: Split Criteria & Variance Reduction](docs/099_regression_trees.md) |
| 100 | [Decision Tree Visualization & Internal Structure Traversal](docs/100_awesome_decision_tree_visualization_using_dtreeviz.md) |
| 101 | [Ensemble Learning & Wisdom of Crowds](docs/101_introduction_to_ensemble_learning.md) |
| 102 | [Voting Ensemble Classifier (Hard vs. Soft Voting)](docs/102_voting_ensemble.md) |
| 103 | [Voting Ensemble Code Demo & Custom Aggregation](docs/103_voting_ensemble.md) |
| 104 | [Voting Regressor](docs/104_voting_ensemble.md) |
| 105 | [Bagging (Bootstrap Aggregation) Intuition](docs/105_bagging.md) |
| 106 | [Bagging Classifier Code Demo](docs/106_bagging_ensemble.md) |
| 107 | [Out-of-Bag (OOB) Evaluator](docs/107_bagging_ensemble.md) |
| 108 | [Random Forest Subspace Feature Sampling](docs/108_introduction_to_random_forest.md) |
| 109 | [Variance Reduction & Tree Decorrelation](docs/109_how_random_forest_performs_so_well.md) |
| 110 | [Random Forest vs Bagging & Gini MDI](docs/110_bagging_vs_random_forest.md) |
| 111 | [Random Forest Hyperparameters](docs/111_random_forest_hyper-parameters.md) |
| 112 | [Hyperparameter Tuning Random Forest using GridSearchCV](docs/112_hyperparameter_tuning_random_forest_using_gridsearchcv.md) |
| 113 | [Out-of-Bag (OOB) Score](docs/113_oob_score.md) |
| 114 | [Feature Importance using Random Forest](docs/114_feature_importance_using_random_forest_and.md) |
| 115 | [How AdaBoost Classifier Works](docs/115_how_adaboost_classifier_works.md) |
| 116 | [AdaBoost Code Demo](docs/116_adaboost.md) |
| 117 | [AdaBoost Algorithm Mechanics](docs/117_adaboost_algorithm.md) |
| 118 | [AdaBoost Hyperparameters and Shrinkage](docs/118_adaboost_hyperparameters.md) |
| 119 | [Bagging vs Boosting](docs/119_bagging_vs_boosting.md) |
| 120 | [Gradient Boosting Regression (Squared Loss)](docs/120_gradient_boosting_explained.md) |
| 121 | [Gradient Boosting Regression - Leaf Optimization](docs/121_gradient_boosting_regression_part_2.md) |
| 122 | [Gradient Boosting for Classification](docs/122_gradient_boosting_for_classification.md) |
| 123 | [Introduction to XGBoost](docs/123_introduction_to_xgboost.md) |
| 124 | [XGBoost for Regression](docs/124_xgboost_for_regression.md) |
| 125 | [XGBoost for Classification](docs/125_xgboost_for_classification.md) |
| 126 | [The Maths Behind XGBoost](docs/126_the_maths_behind_xgboost.md) |
| 127 | [Stacking and Blending Ensembles](docs/127_stacking_and_blending_ensembles.md) |

### Unsupervised Learning & Clustering

| # | Topic |
| :--- | :--- |
| 128 | [K-Means Clustering Algorithm](docs/128_k-means_clustering_algorithm.md) |
| 129 | [Silhouette Analysis & Elbow Method](docs/129_k-means_clustering_algorithm_in_python.md) |
| 130 | [K-Means Clustering Algorithm from Scratch](docs/130_k-means_clustering_algorithm_from_scratch_in.md) |
| 131 | [Agglomerative Hierarchical Clustering](docs/131_agglomerative_hierarchical_clustering.md) |
| 132 | [DBSCAN Clustering Algorithms](docs/132_dbscan_clustering_algorithms.md) |

### Advanced ML Techniques

| # | Topic |
| :--- | :--- |
| 133 | [Imbalanced Data in Machine Learning](docs/133_imbalanced_data_in_machine_learning.md) |
| 134 | [Hyperparameter Tuning Using Optuna](docs/134_hyperparameter_tuning_using_optuna.md) |

---

## 🧑‍💻 Contributing & Feedback

Contributions are welcome! If you find a typo, want to improve explanations, or add extra code examples:
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -m 'Improve topic X'`
4. Push to the branch: `git push origin feature/improvement`
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
