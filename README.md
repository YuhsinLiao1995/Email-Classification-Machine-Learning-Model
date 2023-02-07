# Email-Classification-Machine-Learning-Model


FOUNDATIONS OF MACHINE LEARNING

M.Sc in Data Sciences and Business Analytics
CentraleSupélec

Assignment 2 - Kaggle Challenge


Group members: Besnard Clara, Certo Lucrezia, Granger Leopold, Liao Yuhsin

Group name on Kaggle: MLwithmeTonight

## Feature Engineering

As we first approached the dataset, we wanted to have a general idea of its characteristics. Our initial findings were:
Out of 13 variables: 4 were of the object type (date, org, tld and mail_type) , 5 numerical “continuous”, 3 one hot encoded and lastly there was the “label” target variable of discrete type.
Most of the variables did not have any null values (9/13). The variables with the most number of nulls were the training set specifically org and tld with approximately 3k missing values, while mail_type had less than 200 and chars_in_subject only 16. 
There were some rows that were exact duplicates with the exception of the label.
There was an unequal distribution of labels, with most observations being labeled 1.

From these results we decided that: 
The date variable should be transformed into a date type so to be able to extract more specific information like the month or the day of the week the email was sent. 
Considering that the aim is to run a classification model on the data, the categorical variables will have to go through some form of encoding.  
We needed to investigate the distributions of the numerical variables to understand whether some transformation was needed.
There were few NAs (4% max of the total rows). We decided not to delete them as they still hold information and instead replaced them with either 0 in numerical cases or “not defined” in categorical variables. 
We needed to do something to tackle the duplicates issue.

### a. Date transformation: 
After transforming the date into a datetime type, we decided to split this variable into 5 new variables: day, hour, year, month and weekday. We did this as we believed there might be some patterns regarding the moment the email has been sent and the type of email. For example updates from the bank/insurance might be sent regularly on a specific day of the week.
Used variables: date (object type)
Generated variables: day, hour, year, month and weekday (float type)

### b. Encoding: 
we performed two types of encoding on the object variables. We did one hot for the mail_type because the main information we wanted to focus on was the type associated with the email. The  mail_type values were strings with all the types concatenated, so to properly encode them we first had to split the string into its composing parts and then perform the encoding on those results.   Regarding the org and tdl variables, we performed target-encoding based on each category’s probability to target labels. For example, if the category “facebook” appears in 100 rows, 50 with label 1 and 50 with  label 2, we will create the columns “org_label_1” = 0.5 and “org_label_2” = 0.5. To do this we created a table storing all the categories and their corresponding probability to each label. We then merged it to both the training and testing dataframes. By doing this, we discovered that some categories existed in the test set but not in the training one (and vice versa), however since the number of such occurrences was limited, we decided to just append the variables’ columns and  fill them with zeros.
Used variables: org, tld, mail_type (object type)
Generated variables: multipart, text, alternative etc. (float types). orgx_label_1, orgx_label_2 , orgx_label_3, orgx_label_4 etc. (float types). tldx_label_1, tldx_label_2, tldx_label_3, tldx_label_4 etc. (float types)
  
### c. Variables distribution: 
By looking at the distribution of the numerical variables we noticed that: 1) all the variables seem to be strongly positively skewed with a peak in frequency around 0. 2) The presence of outliers strongly biased upward the mean (Exhibit A), and that this characteristic would bias any estimation if not dealt with. From this we decided that the most important piece of information we needed to retain from these variables was: whether the variable is equal to zero or not, and a general idea of the magnitude of the value. In our opinion a good way to achieve this was to encode these variables based on the quantile they belonged to. Therefore we associated each value to one of 6 bins where: 0-20% → 1, 20-40% → 2, 40-60% → 3 etc. Plus a bin 0 where all the zero values were entered (applied to all the numerical variables with the exception of “chars_in_body” as it did not have any 0 values). 
Used variables: chars_in_body, chars_in_subject, images, urls (int type)
Generated variables: quantile_chars_body, quantile_chars_subjec , quantile_images, quantile_urls (int type) 

### Additional engineering: 
We also performed an additional layer of feature engineering on the tld and org variables. By studying the values of these variables, we noticed that some of them were linked to a certain type of institution ( ex. “centralesupelec” and “iiit” are academic institutions). Because of this, we thought that having variables marking the type of institution the sender belonged to could increase the performance of the model. We were able to identify four types of institutions: academic, government, e-learning and travel. We then created 4 new variables where we hot encoded all the observations that satisfied the belongingness criteria of each type of institution.  
Used variables: org, tld (object type)
Generated variables: academic, government, elearning, traveling (int type)

Furthermore, as stated earlier, some rows are exact duplicates with the exception of the label. Hence, one email can get more than one label. In addition, some of the emails in the training set were also present in the test set. In order to take this information into account, we created 8 new columns, each corresponding to a label, initialized at 0. For each row, using a dictionary, we checked whether the email had duplicates and their corresponding label. We would then set the value of column matching to the duplicate’s label to 1. It is important to note that the column associated with the current label is always set to 0. For the test set, we created a new column label, initialized with 9 (any value is correct except 0-7), and we applied the same logic. A more schematised explanation is available in Exhibit B.
Generated variables: label_0, label_1, label_2 etc. (int type)

Feature combination: Additionally, we also decided to explore feature combination techniques. In particular, we explored the combination of the number of characters and the number of images. We found two ways to investigate this combination, first, taking the max value between the quantile of characters and the quantile of images. This gives us information on whether an email is more verbose than image heavy. Second, we took a log transform of the number of characters and the number of images (log(n char) + log (n images)). The intuition behind this second transformation is that the log transform is known to reduce the skewness of a measurement variable. More precisely, we wanted to capture the fact that an email with zero characters is quite different from an email with 1 character, whereas an email with 32 characters is similar in nature to an email with 33. The same logic applies to the log transform of the number of images. 
Used variables: quantile_chars_body, quantile_image (int type)
Generated variables:  image_heavy (int type), log_image_char (float type)

Feature selection: As a result of our feature engineering, we had hundreds of variables, with little indication of which ones were statistically relevant. Considering the risk of overfitting, it was necessary for us to select the features that would generalize well.  Our feature selection method was to run a Xgboost model on the whole dataset, then we extracted the features importance and retained only those features with an importance greater than 0.001 (circa 10% of the total number).  Before using Xgboost, we also performed feature selection using Lasso Regression and Mutual Information, however they yielded unsatisfactory results in comparison to the final method. 
Model Tuning and Comparison
After feature engineering, we explored different models to evaluate the performance. We splitted the training set with test size = 0.25.

#### KNN: 
We started with KNN as our baseline model. We trained KNN with n_neighbors = 3 and gained the accuracy rate = 0.4284 on our test set. Then we tuned the n_neighbors to evaluate which model performed the best. We discovered that when n_neighbors = 30, the accuracy reached 0.5152, then even if n_neighnor increased, the change of variance was not significant. Then we ran a Lasso model to find the significant features. The Lasso model selected 29 variables and eliminated 19. We  then ran the KNN model again with these variables, gaining a 0.5155 accuracy rate. Then we performed cross validation, we ended up with 0.51993 accuracy rate on the kaggle test set.

#### RandomForest: 
We trained the model on all features, and after tuning with RandomizedSearchCV, we obtained a score of 0.601 (computed through cross-validation) with feature selection based on XG boost. On the test set, we got a score of 0.5983, which was promising.

#### SVM: 
We trained the SVM model with gamma = “auto”, and we gained an accuracy rate of 0.5975. This accuracy rate performs well. Then we selected the features with top importance then submitted our prediction on the testing set. In the end we gained 0.57443

#### XGBoosting: 
At the beginning, we trained the XGBoost model without tuning parameters. We gained 0.605 from the first try. Then we used the feature_importances_ function from XGBoost to select variables with importance greater than 0.001. Afterwards, we performed RandomizedSearchCV to help us tune the parameters and applied cross validation on our model to prevent over-fitting. We discovered the best parameters: colsample_bytree=0.7, gamma=0.4, learning_rate=0.05, max_depth=6, min_child_weight=5, objective='multi:softprob' which helped us we gained 0.6102 on our own testing data set and 0.598 on the kaggle test set, making this our best performing model. We then dealt with the duplicates as explained earlier in this report, which greatly contributed to increasing our score. 

#### Cross validated performance: 
We evaluate the model by using repeated classified k fold cross-validation, with three repeats and 10 folds. We used accuracy to measure performance.


Model | Train Set Score after tuning | Test Set Score | Comments |
--- | --- | --- | --- |
KNN | 0.5155 | 0.5199 | Baseline model | 
XGBoost | 0.5155 | 0.5199 | Best one | 
SVM | 0.5975 | 0.5744 | Has potential | 
RandomForest | 0.6026 | 0.5933 | Has potential | 
KNN | 0.5155 | 0.5199 | Baseline model | 

Additional models that have not performed well: Our two first submissions had very low scores: 0.37 and 0.33. Both had very good accuracy scores on the training set (around 0.57) compared to the leaderboard at the beginning of the competition,  but performed poorly on the test set, meaning our models were overfitting. For the first model, we tried autosklearn, an autoML library that builds ensemble models from a list of common classification algorithms. This initially performed well on training data, but generalized poorly.	 Our second model was based on sklearn’s decision tree classifier. 
After these two submissions, we realized that our feature engineering was not complete and that would explain why the model did work as expected. Numerical data were normalized and not discretized, which was not optimal for classifiers like decision trees. Furthermore, we had different feature selection methods at that time: we used the Lasso regression and mutual information. We realized that Lasso was not the best choice for categorical data (yet, the majority of the features are categorical). For the second model, we used Mutual Information, but then, we performed PCA, which again, is not compatible with a dataset with a lot of categorical features. Hence, the models failed because we had not finished our feature engineering and because of poor choice of tools.
We had another very low result (0.25), which is the result of an error in the feature selection code. 
Additional notes: We have noticed that other models performed well compared to the main one. As the duplicate feature was a late addition to our feature engineering, we were not able to train the model with this new data.
Appendix

Exhibit A


Exhibit B - Duplicates emails with different labels

Before engineering:
Some emails are duplicates but have different labels in the train set. 

Email | Label | 
--- | --- |
x1 | 1 | 
x2 | 0 | 
x1 | 3 | 
x4 | 2 | 
x4 | 4 | 


and some email are also present in the test set: 

Email |
--- |
x45 |
x9 |
x1 |
x76 | 
x13 | 
x4 |


We thus create a dictionary listing all labels for each email. 
dup_dic = {
x1 = [1, 3],
x2 = [0],
x3 = [0],
x4 = [2, 4]
}

After engineering: 
Train set: 

Email | Label | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | 
--- | --- | --- | --- | --- | --- | --- |
x1 | 1 | 0 | 0 | 0 | 1 | 0 |
x2 | 0 | 0 | 0 | 0 | 0 | 0 |
x1 | 3 | 0 | 1 | 0 | 0 | 0 | 
x3 | 0 | 0 | 0 | 0 | 0 | 0 | 
x4 | 2 | 0 | 0 | 0 | 0 | 1 |
x4 | 4 | 0 | 0 | 1 | 0 | 0 |



Test set:

Email | Label | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | 
--- | --- | --- | --- | --- | --- | --- |
x45 | 9 | 0 | 0 | 0 | 0 | 0 |
x9 | 9 | 0 | 0 | 0 | 0 | 0 |
x1 | 9 | 0 | 1 | 0 | 1 | 0 | 
x76 | 9 | 0 | 0 | 0 | 0 | 0 | 
x13 | 9 | 0 | 0 | 0 | 0 | 0 |
x4 | 9 | 0 | 0 | 1 | 0 | 1 |








