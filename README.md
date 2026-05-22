# 🚀 100+ Days of Machine Learning

Welcome to the **100+ Days of Machine Learning** repository! This is a comprehensive, day-by-day structured curriculum covering everything from foundational ML concepts, preprocessing, and exploratory data analysis (EDA), to supervised regression, classification, advanced ensemble methods, and unsupervised clustering.

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

Here is a structured index of the daily notes. Click on any topic to view the detailed study guide.

### Foundations & Setup (Days 1-14)

| Day | Topic |
| :--- | :--- |
| Day 001 | [What is Machine Learning?](001_what_is_machine_learning.md) |
| Day 002 | [AI vs. ML vs. DL](002_ai_vs_ml_vs_dl_for.md) |
| Day 003 | [Types of Machine Learning](003_types_of_machine_learning_for_beginners.md) |
| Day 004 | [Batch Machine Learning (Offline Learning)](004_batch_machine_learning.md) |
| Day 005 | [Online Machine Learning (Incremental Learning)](005_online_machine_learning.md) |
| Day 006 | [Instance-based vs. Model-based Learning](006_instance-based_vs_model-based_learning.md) |
| Day 007 | [Challenges in Machine Learning](007_challenges_in_machine_learning.md) |
| Day 008 | [Real-World Applications of Machine Learning](008_application_of_machine_learning.md) |
| Day 009 | [Machine Learning Development Life Cycle (MLDLC)](009_machine_learning_development_life_cycle.md) |
| Day 010 | [Data Roles - Data Engineer vs. Data Analyst vs. Data Scientist vs. ML Engineer](010_data_engineer_vs_data_analyst_vs.md) |
| Day 011 | [What are Tensors?](011_what_are_tensors.md) |
| Day 012 | [Installing Anaconda & Development Environments](012_installing_anaconda_&_colab_setup.md) |
| Day 013 | [End-to-End Machine Learning Toy Project](013_end-to-end_toy_project.md) |
| Day 014 | [Framing Machine Learning Problems](014_framing_ml_problems.md) |

### Data Preprocessing & EDA (Days 15-49)

| Day | Topic |
| :--- | :--- |
| Day 015 | [Working with CSV Files in Pandas](015_working_with_csv_files_in_pandas.md) |
| Day 016 | [Working with JSON & SQL in Pandas](016_working_with_json_sql.md) |
| Day 017 | [Fetching Data From an API](017_fetching_data_from_an_api.md) |
| Day 018 | [Fetching Data Using Web Scraping](018_fetching_data_using_web_scraping.md) |
| Day 019 | [Understanding Your Data](019_understanding_your_data.md) |
| Day 020 | [Exploratory Data Analysis (EDA) - Univariate Analysis](020_eda_using_univariate_analysis.md) |
| Day 021 | [EDA using Bivariate & Multivariate Analysis](021_eda_using_bivariate_and_multivariate_analysis.md) |
| Day 022 | [Automated EDA with Pandas Profiling (`ydata-profiling`)](022_pandas_profiling.md) |
| Day 023 | [What is Feature Engineering?](023_what_is_feature_engineering.md) |
| Day 024 | [Feature Scaling - Standardization](024_feature_scaling.md) |
| Day 025 | [Feature Scaling - Normalization](025_feature_scaling.md) |
| Day 026 | [Encoding Categorical Data (Ordinal Encoding & Label Encoding)](026_encoding_categorical_data.md) |
| Day 027 | [One-Hot Encoding (OHE)](027_one_hot_encoding.md) |
| Day 028 | [ColumnTransformer in Machine Learning](028_column_transformer_in_machine_learning.md) |
| Day 029 | [Machine Learning Pipelines A–Z](029_machine_learning_pipelines_a-z.md) |
| Day 030 | [Function Transformer & Mathematical Transforms](030_function_transformer.md) |
| Day 031 | [Power Transformer: Box-Cox & Yeo-Johnson Transformations](031_power_transformer.md) |
| Day 032 | [Discretization: Binning & Binarization](032_binning_and_binarization.md) |
| Day 033 | [Handling Mixed Variables](033_handling_mixed_variables.md) |
| Day 034 | [Handling Date & Time Variables](034_handling_date_and_time_variables.md) |
| Day 035 | [Handling Missing Data Part 1: Complete Case Analysis (CCA)](035_handling_missing_data.md) |
| Day 036 | [Handling Missing Data Part 2: Numerical Mean/Median/Arbitrary Imputation](036_handling_missing_data.md) |
| Day 037 | [Handling Missing Data Part 3: Categorical Imputation](037_handling_missing_categorical_data.md) |
| Day 038 | [Missing Indicator & Random Sample Imputation](038_missing_indicator.md) |
| Day 039 | [K-Nearest Neighbors Imputer (KNNImputer)](039_knn_imputer.md) |
| Day 040 | [Multivariate Imputation by Chained Equations (MICE)](040_multivariate_imputation_by_chained_equations_for.md) |
| Day 041 | [What are Outliers?](041_what_are_outliers.md) |
| Day 042 | [Outlier Detection & Removal Using Z-Score](042_outlier_detection_and_removal_using_z-score.md) |
| Day 043 | [Outlier Detection & Removal Using the Interquartile Range (IQR) Method](043_outlier_detection_and_removal_using_the.md) |
| Day 044 | [Outlier Detection Using the Percentile Method](044_outlier_detection_using_the_percentile_method.md) |
| Day 045 | [Feature Construction & Feature Splitting](045_feature_construction.md) |
| Day 046 | [The Curse of Dimensionality](046_curse_of_dimensionality.md) |
| Day 047 | [Principal Component Analysis (PCA): Geometric Intuition](047_principle_component_analysis_pca.md) |
| Day 048 | [Principal Component Analysis (PCA): Mathematical Derivation](048_principle_component_analysis_pca.md) |
| Day 049 | [Practical PCA in Scikit-Learn](049_principle_component_analysis_pca.md) |

### Supervised Learning: Regression (Days 50-69)

| Day | Topic |
| :--- | :--- |
| Day 050 | [Simple Linear Regression (OLS closed-form)](050_simple_linear_regression.md) |
| Day 051 | [Simple Linear Regression: Geometric Interpretation & Custom Implementation](051_simple_linear_regression.md) |
| Day 052 | [Regression Evaluation Metrics: Mathematical Derivations & Code Demo](052_regression_metrics.md) |
| Day 053 | [Multiple Linear Regression: Geometric Intuition & Scikit-Learn Implementation](053_multiple_linear_regression.md) |
| Day 054 | [Multiple Linear Regression: Mathematical Derivation of the Normal Equation](054_multiple_linear_regression.md) |
| Day 055 | [Multiple Linear Regression: Coding the Normal Equation from Scratch](055_multiple_linear_regression.md) |
| Day 056 | [Assumptions of Linear Regression & Diagnostic Tests in Python](056_what_are_the_main_assumptions_of.md) |
| Day 057 | [Gradient Descent Optimization from Scratch](057_gradient_descent_from_scratch.md) |
| Day 058 | [Vectorized Batch Gradient Descent for Multiple Linear Regression](058_batch_gradient_descent_with_code_demo.md) |
| Day 059 | [Stochastic Gradient Descent (SGD)](059_stochastic_gradient_descent.md) |
| Day 060 | [Mini-Batch Gradient Descent](060_mini-batch_gradient_descent.md) |
| Day 061 | [Polynomial Regression: Mathematical Formulation & Pipeline Implementation](061_polynomial_regression.md) |
| Day 062 | [The Bias-Variance Trade-off: Mathematical Decomposition & Simulation](062_bias_variance_trade-off.md) |
| Day 063 | [Ridge Regression (L2 Regularization): Intuition & Coefficient Shrinkage](063_ridge_regression_part_1.md) |
| Day 064 | [Ridge Regression: Closed-form Mathematical Matrix Derivation](064_ridge_regression_part_2.md) |
| Day 065 | [Ridge Regression: Geometric Interpretation & Lagrange Multipliers](065_ridge_regression_part_3.md) |
| Day 066 | [5 Key Points on Regularization: Theory & Practical Implementation](066_5_key_points.md) |
| Day 067 | [Lasso Regression (L1 Regularization): Cost Function & Feature Selection](067_lasso_regression.md) |
| Day 068 | [Why Lasso Regression Creates Sparsity: Geometry & Coordinate Descent](068_why_lasso_regression_creates_sparsity.md) |
| Day 069 | [Elastic Net Regression: Combining L1 and L2 Regularizations](069_elasticnet_regression.md) |

### Supervised Learning: Classification (Days 70-96)

| Day | Topic |
| :--- | :--- |
| Day 070 | [Introduction to Logistic Regression: Odds, Log-Odds, & The Sigmoid Function](070_logistic_regression_part_1.md) |
| Day 071 | [The Perceptron Trick: Binary Classification from a Geometric Perspective](071_logistic_regression_part_2.md) |
| Day 072 | [Logistic Regression Part 3: Deep Dive into Probability, Odds, Log-Odds, and Sigmoid Symmetry](072_logistic_regression_part_3.md) |
| Day 073 | [Logistic Regression Part 4: Why MSE Fails, Likelihood, & Binary Cross-Entropy Loss](073_logistic_regression_part_4.md) |
| Day 074 | [Calculus of the Sigmoid Function: Derivation & Numerical Verification](074_derivative_of_sigmoid_function.md) |
| Day 075 | [Logistic Regression Part 5: Gradient Derivation & Gradient Descent from Scratch](075_logistic_regression_part_5.md) |
| Day 076 | [Model Evaluation: Accuracy & The Confusion Matrix](076_accuracy_and_confusion_matrix.md) |
| Day 077 | [Class Imbalance Metrics: Precision, Recall, & F1-Score](077_precision_recall_and_f1_score.md) |
| Day 078 | [Model Evaluation: The ROC Curve & AUC](078_roc_curve_in_machine_learning.md) |
| Day 079 | [Softmax Regression (Multinomial Logistic Regression)](079_softmax_regression.md) |
| Day 080 | [Non-Linear Boundaries: Polynomial Logistic Regression](080_polynomial_features_in_logistic_regression.md) |
| Day 081 | [Logistic Regression Hyperparameters: Regularization & Solvers](081_logistic_regression_hyperparameters.md) |
| Day 082 | [Naive Bayes Classifier Part 1: Conditional Probability](082_naive_bayes_classifier.md) |
| Day 083 | [Naive Bayes Classifier Part 2: Independent Events](083_naive_bayes_classifier.md) |
| Day 084 | [Naive Bayes Classifier Part 3: Mutually Exclusive Events](084_naive_bayes_classifier.md) |
| Day 085 | [Naive Bayes Classifier Part 4: Derivation of Bayes' Theorem](085_naive_bayes_classifier.md) |
| Day 086 | [Naive Bayes Classifier Part 5: The Law of Total Probability](086_naive_bayes_classifier.md) |
| Day 087 | [Naive Bayes Classifier Part 6: The "Naive" Assumption](087_naive_bayes_classifier.md) |
| Day 088 | [Naive Bayes Classifier Part 7: Log Likelihood & Laplace Smoothing](088_naive_bayes_classifier.md) |
| Day 089 | [Naive Bayes Classifier Part 8: Bag-of-Words Text Classification](089_naive_bayes_classifier.md) |
| Day 090 | [Naive Bayes Classifier Part 9: Gaussian Naive Bayes](090_naive_bayes_part_9.md) |
| Day 091 | [K-Nearest Neighbors (KNN) Classifier](091_what_is_k_nearest_neighbors.md) |
| Day 092 | [Support Vector Machines (Geometric Intuition)](092_support_vector_machines.md) |
| Day 093 | [Mathematics of SVM - Primal Formulation](093_mathematics_of_svm.md) |
| Day 094 | [Mathematics of Support Vector Machine: Dual Formulation & KKT Conditions](094_mathematics_of_support_vector_machine.md) |
| Day 095 | [Mathematics of SVM: Non-linear Mapping & The Kernel Trick](095_kernel_trick_in_svm.md) |
| Day 096 | [SVM Kernel Functions & Hyperparameter Tuning](096_kernel_trick_in_svm.md) |

### Tree-Based & Ensemble Methods (Days 97-127)

| Day | Topic |
| :--- | :--- |
| Day 097 | [Decision Trees: Geometric Intuition & Space Partitioning](097_decision_trees_geometric_intuition.md) |
| Day 098 | [Decision Trees: Split Metrics & Hyperparameter Tuning](098_decision_trees.md) |
| Day 099 | [Regression Trees: Split Criteria & Variance Reduction](099_regression_trees.md) |
| Day 100 | [Decision Tree Visualization & Internal Structure Traversal](100_awesome_decision_tree_visualization_using_dtreeviz.md) |
| Day 101 | [Ensemble Learning & Wisdom of Crowds](101_introduction_to_ensemble_learning.md) |
| Day 102 | [Voting Ensemble Classifier (Hard vs. Soft Voting)](102_voting_ensemble.md) |
| Day 103 | [Voting Ensemble Code Demo & Custom Aggregation](103_voting_ensemble.md) |
| Day 104 | [Voting Regressor](104_voting_ensemble.md) |
| Day 105 | [Bagging (Bootstrap Aggregation) Intuition](105_bagging.md) |
| Day 106 | [Bagging Classifier Code Demo](106_bagging_ensemble.md) |
| Day 107 | [Out-of-Bag (OOB) Evaluator](107_bagging_ensemble.md) |
| Day 108 | [Random Forest Subspace Feature Sampling](108_introduction_to_random_forest.md) |
| Day 109 | [Variance Reduction & Tree Decorrelation](109_how_random_forest_performs_so_well.md) |
| Day 110 | [Random Forest vs Bagging & Gini MDI](110_bagging_vs_random_forest.md) |
| Day 111 | [Random Forest Hyperparameters](111_random_forest_hyper-parameters.md) |
| Day 112 | [Hyperparameter Tuning Random Forest using GridSearchCV](112_hyperparameter_tuning_random_forest_using_gridsearchcv.md) |
| Day 113 | [Out-of-Bag (OOB) Score](113_oob_score.md) |
| Day 114 | [Feature Importance using Random Forest](114_feature_importance_using_random_forest_and.md) |
| Day 115 | [How AdaBoost Classifier Works](115_how_adaboost_classifier_works.md) |
| Day 116 | [AdaBoost Code Demo](116_adaboost.md) |
| Day 117 | [AdaBoost Algorithm Mechanics](117_adaboost_algorithm.md) |
| Day 118 | [AdaBoost Hyperparameters and Shrinkage](118_adaboost_hyperparameters.md) |
| Day 119 | [Bagging vs Boosting](119_bagging_vs_boosting.md) |
| Day 120 | [Gradient Boosting Regression (Squared Loss)](120_gradient_boosting_explained.md) |
| Day 121 | [Gradient Boosting Regression - Leaf Optimization](121_gradient_boosting_regression_part_2.md) |
| Day 122 | [Gradient Boosting for Classification](122_gradient_boosting_for_classification.md) |
| Day 123 | [Introduction to XGBoost](123_introduction_to_xgboost.md) |
| Day 124 | [XGBoost for Regression](124_xgboost_for_regression.md) |
| Day 125 | [XGBoost for Classification](125_xgboost_for_classification.md) |
| Day 126 | [The Maths Behind XGBoost](126_the_maths_behind_xgboost.md) |
| Day 127 | [Stacking and Blending Ensembles](127_stacking_and_blending_ensembles.md) |

### Unsupervised Learning & Clustering (Days 128-132)

| Day | Topic |
| :--- | :--- |
| Day 128 | [K-Means Clustering Algorithm](128_k-means_clustering_algorithm.md) |
| Day 129 | [Silhouette Analysis & Elbow Method](129_k-means_clustering_algorithm_in_python.md) |
| Day 130 | [K-Means Clustering Algorithm from Scratch](130_k-means_clustering_algorithm_from_scratch_in.md) |
| Day 131 | [Agglomerative Hierarchical Clustering](131_agglomerative_hierarchical_clustering.md) |
| Day 132 | [DBSCAN Clustering Algorithms](132_dbscan_clustering_algorithms.md) |

### Advanced ML Techniques (Days 133-134)

| Day | Topic |
| :--- | :--- |
| Day 133 | [Imbalanced Data in Machine Learning](133_imbalanced_data_in_machine_learning.md) |
| Day 134 | [Hyperparameter Tuning Using Optuna](134_hyperparameter_tuning_using_optuna.md) |

---

## 🧑‍💻 Contributing & Feedback

Contributions are welcome! If you find a typo, want to improve explanations, or add extra code examples:
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -m 'Improve day X note'`
4. Push to the branch: `git push origin feature/improvement`
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
