# LATAM Challenge Documentation

by Sebastian Cáceres Gelvez

## Part I

### 1. Model Selection

Model selection consisted of the following steps:

* First, I started by reviewing the `exploration.ipynb` notebook and 
the analysis made by the DS.

* While going through the notebook, I checked the `data` DataFrame every
often to see how data was being handled/modified and to understand how it 
was input to the model in the form of `features`.

* In section **4. Training**, I went a bit further and produced
a *Normalized Confusion Matrix* plot to more easily identify how each 
model was performing. I repeated this step for the improved models in 
section **6. Training with Improvement**.

* I also saved the output from the `classification_report()` method for 
each model and added the computation of the `roc_auc_score()` metric to it. 
The information from the classification report allowed me to generate a 
summary table with the performance metrics for the six proposed models, which
is shown in section **8. Performance Summary** and below:

    - First, I defined the `create_classif_summary()` function to convert the 
    classification reports for all the models into a `df_summary` DataFrame.
    
    - Then, using this method, I input the classification reports and 
    generated a summary table for their performance metrics. Below are some
    conventions:

        `xgb`: Simple XGBoost

        `lreg`: Simple Logistic Regression

        `xgb2-bal-topfeat`: XGBoost with Feature Importance and Balanced Data

        `xgb3-nobal-topfeat`: XGBoost with Feature Importance but *without* Balanced Data

        `lreg2-bal-topfeat`: Logistic Regression with Feature Importance and Balanced Data

        `lreg3-nobal-topfeat`: Logistic Regression with Feature Importance but *without* 
        Balanced Data

        |                |      xgb | xgb2-bal-topfeat | xgb3-nobal-topfeat |      lreg | lreg2-bal-topfeat  | lreg3-nobal-topfeat |
        |:---------------|---------:|-----------------:|-------------------:|----------:|-------------------:|--------------------:|
        | 0_precision    | 0.812778 |        0.879198  |         0.813651   | 0.816599  |          0.878345  |           0.814335  |
        | 1_precision    | 0        |        0.249248  |         0.764706   | 0.558952  |          0.247715  |           0.529412  |
        | 0_recall       | 1        |        0.522357  |         0.999563   | 0.994479  |          0.518585  |           0.997376  |
        | 1_recall       | 0        |      **0.68842** |         0.00616991 | 0.0303749 |        **0.688182**|           0.0128144 |
        | accuracy       | 0.812778 |        0.553448  |         0.813577   | 0.813977  |          0.550338  |           0.813044  |
        | 0_f1-score     | 0.896721 |        0.655351  |         0.897076   | 0.896803  |          0.65214   |           0.896609  |
        | 1_f1-score     | 0        |      **0.365988**|         0.0122411  | 0.0576187 |        **0.364299**|           0.0250232 |
        | w-avg_f1-score | 0.728835 |        0.601176  |         0.731415   | 0.739689  |          0.598249  |           0.733429  |
        | roc-auc-score  | 0.5      |      **0.605388**|         0.502866   | 0.512427  |        **0.603384**|           0.505095  |

* As the DS mentioned in the conclusions, the summary table confirmed that 
there is not much difference between XGBoost and Logistic Regression models, 
and the use of balanced classes improved the performance of the models in terms 
of `1_recall`, `1_f1-score` and `roc-auc-score`. However, the XGBoost model 
`xgb2-bal-topfeat` **performed slightly better** than the Logistic Regression model 
`lreg2-bal-topfeat`.

* Thus, the **final decision was to use the `xgb2-bal-topfeat` model**, i.e., the 
*XGBoost with Feature Importance and Balanced Data* model

### 2. Model Transcription

Model transcription was carried out successfully. Two additional methods were 
included in the `DelayModel` class for the *preprocessing* step:

* `get_min_diff()`: Gets minute difference between scheduled (`'Fecha-I'`) and
operation dates (`'Fecha-O'`).

* `get_delay()`: Gets delay column (target) in case the argument `target_column`
is given when calling the `preprocess()` method, but it is not a column in the 
raw data.
