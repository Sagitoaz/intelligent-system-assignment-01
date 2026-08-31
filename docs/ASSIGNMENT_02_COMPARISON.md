# Assignment 02 Cross-Application Comparison

## Representation summary

| Application | Raw form | Numerical representation | Model input shape |
|---|---|---|---|
| Diabetes | CSV numeric patient record | Median-imputed and standardized six-feature vector | `B × 6` |
| House Price | CSV mixed property listing | Median/mode imputation, numeric scaling, categorical one-hot encoding | `B × 83` |
| Customer Preference | CSV review text + helpfulness counts | TF-IDF (`V=12,000`) + five engineered numerical values | `B × (12,000 + 5) = B × 12,005` |

`B` is the number of observations presented at once. Diabetes `d=6`. House `d_raw=11` expands to `d_encoded=83`. Ecommerce `V=12,000`, `d_tabular=5`, and final `d=12,005`.

## System comparison

| Aspect | Diabetes | House Price | Customer Preference |
|---|---|---|---|
| Problem type | Binary classification | Regression | Binary text classification |
| One observation | One patient record | One property listing | One customer-product review |
| Target | Outcome 0/1 | Price, billion VND | Score-derived negative/positive |
| Raw representation | 6 numeric fields | 11 numeric/categorical fields | 2 text + 2 count fields |
| Final representation | Scaled dense vector | Encoded dense vector | High-dimensional sparse vector |
| Raw feature count | 6 | 11 | 4 deployable fields |
| Encoded dimension | 6 | 83 | 12,005 |
| Data-quality issues | Physiologically impossible hidden zeros | Missing/mixed listing values, address-derived province | Missing summary, neutral scores, duplicates, invalid helpfulness, imbalance |
| Preprocessing | Hidden-zero handling, median, scaling | Province derivation, imputation, scaling, one-hot | Text combine, TF-IDF, engineered lengths/ratio, max-absolute scaling |
| Best persisted model | Random Forest Classifier | Random Forest Regressor | Logistic Regression |
| Main metric | Recall/F1 with accuracy context | RMSE/MAE and R² | F1 and ROC-AUC |
| Training time | Existing A1 experiment; unchanged | A2 Ridge 0.65 s; Gradient Boosting 13.63 s (3-fold CV wall time) | final refit 22.82 s; six-model validation 0.05–47.38 s |
| Persistence | Complete joblib Pipeline | Complete joblib Pipeline | Complete joblib Pipeline + importable transformers |
| Web | `/diabetes` | `/house-price` | `/ecommerce` |
| Mobile | Diabetes | House Price | E-commerce |
| Main limitation | Small observational dataset; not clinical | Listings are not transactions | Rating proxy, imbalance, near/user/product dependence |

## Discussion questions

1. The datasets differ in domain, scale, modality and target: 768 health rows, 30,229 mixed listings, and 568,454 text reviews.
2. An observation means a patient, listing, or customer-product review respectively.
3. A raw column is a measured attribute, listing property, or review/count field; encoded columns can instead be learned categories or vocabulary terms.
4. Diabetes uses a binary observed label, House a continuous price, and Ecommerce a binary proxy derived from ordinal Score.
5. Classification needs precision/recall/F1/ROC-AUC; regression error has magnitude, so MAE/RMSE/R² are appropriate.
6. House categorical columns require one-hot encoding; Ecommerce text requires TF-IDF. Diabetes has no categorical input.
7. Diabetes numeric values and House numeric branches need scaling for scale-sensitive comparisons. Ecommerce engineered counts use max-absolute scaling.
8. Scaling loses original units, one-hot loses category proximity, and TF-IDF loses most word order/context and identifiers.
9. Relative numeric information, explicit categories, and weighted unigram/bigram lexical evidence are preserved.
10. Leakage occurs if preprocessors see test data, Province derivation uses target, Score/target enters Ecommerce X, or duplicates cross splits. The implemented Pipelines and cleaning boundaries prevent these direct cases.
11. Persisted best models are Random Forest for Diabetes/House and Logistic Regression for Ecommerce.
12. Existing A1 selections remain unchanged to avoid test-driven reselection. Ecommerce uses a predeclared validation rule: best probability-capable model within 0.01 of overall-best F1, ROC-AUC tie-break.
13. Diabetes is easiest to deploy because it has six numeric inputs and a small artifact.
14. Ecommerce is most computationally demanding because 120,000 modeling rows produce a 12,005-dimensional sparse matrix across six models.
15. Dataset bias, proxy targets, class imbalance, listing-versus-transaction mismatch, and unseen/near-duplicate dependencies remain.

## Ecommerce evidence

Controlled validation with fixed Logistic Regression:

| Representation | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| A: tabular only | 0.8442 | 0.8445 | 0.9994 | 0.9155 | 0.6410 | 0.24 |
| B: text only | 0.9570 | 0.9627 | 0.9873 | 0.9749 | 0.9836 | 2.25 |
| C: text + tabular | 0.9574 | 0.9631 | 0.9874 | 0.9751 | 0.9840 | 3.25 |

Text materially improves preference prediction; the engineered fields add a small validation gain.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9574 | 0.9631 | 0.9874 | 0.9751 | 0.9840 | 3.48 |
| Decision Tree | 0.8989 | 0.9230 | 0.9604 | 0.9413 | 0.8568 | 47.38 |
| Random Forest | 0.8588 | 0.8567 | 1.0000 | 0.9228 | 0.9512 | 5.56 |
| Linear SVC | 0.9598 | 0.9703 | 0.9825 | 0.9764 | 0.9832 | 3.16 |
| SGD Classifier | 0.9531 | 0.9565 | 0.9894 | 0.9727 | 0.9828 | 0.55 |
| Complement NB | 0.9031 | 0.9847 | 0.8992 | 0.9400 | 0.9707 | 0.05 |

Frozen Logistic Regression test: Accuracy 0.9566, Precision 0.9639, Recall 0.9855, F1 0.9746, ROC-AUC 0.9842; confusion matrix `[[2242, 561], [221, 14976]]`.

House A2 training-only additions: Ridge CV MAE 1.4531, RMSE 1.8206, R² 0.3230 (0.65 s); Gradient Boosting CV MAE 1.3045, RMSE 1.6375, R² 0.4523 (13.63 s). They do not alter the frozen Random Forest selection.
