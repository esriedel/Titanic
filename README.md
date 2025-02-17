# Titanic

The data is from a beginning Kaggle competition to predict survivors of the famous Titanic disaster. Data of 2224 passengers is split into training and test data sets. The challenge is to develop of model from the training data set provided to predict who survived the Titanic disaster. This model is then used to predict whether a passenger survived in the test data set. 

## Initial review of data
* Age has a large number of missing values. Explore with the possibility of replacing the missing values.
* Cabin has a very large set of missing values. 
* Embarked has 2 missing values. 

List of variables
* Survival	0 = No, 1 = Yes
* pclass: Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
* sex	Sex	
* Age	Age in years	
* sibsp	# of siblings / spouses aboard the Titanic	
* parch	# of parents / children aboard the Titanic	
* ticket	Ticket number	
* fare	Passenger fare	
* cabin	Cabin number	
* embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
* pclass: A proxy for socio-economic status (SES) 1st = Upper, 2nd = Middle, 3rd = Lower
* age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
* sibsp: The dataset defines family relations in this way. Sibling = brother, sister, stepbrother, stepsister
* Spouse = husband, wife (mistresses and fiancés were ignored)
* parch: The dataset defines family relations in this way. Parent = mother, father
* Child = daughter, son, stepdaughter, stepson. Some children travelled only with a nanny, therefore parch=0 for them.

### First 10 rows of dataframe

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




### Distributions of variables


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891.000000</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>891</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>681</td>
      <td>NaN</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>347082</td>
      <td>NaN</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>577</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>4</td>
      <td>644</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>NaN</td>
      <td>32.204208</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>NaN</td>
      <td>49.693429</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>7.910400</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>14.454200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>NaN</td>
      <td>512.329200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

| Model | Accuracy | Notes |
| --- | --- | --- |
| logistic regression | 0.76794 | |
| logistic regression w/ tuning| 0.78229 | |
| random forest | 0.76076 | |
| random forest w/ tuning | 0.76794 | |
| k nearest neighbor | 0.77511 | Highest accuracy for K=5 |
| support vector machine | 0.77272 | |
| support vector machine w/ tuning | 0.75358 | |


    
![png](output_4_1.png)
    


   
![png](output_4_3.png)
    



    
![png](output_4_4.png)
    



![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    



    
![png](output_5_4.png)
    



    
![png](output_5_5.png)
    



    
![png](output_5_6.png)
    



```python
missing_values=titanic.isnull().sum()
print(missing_values)
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    


```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


## Explore bivariate relationships and how to handle missing values

First explore whether we should replace missing values of age based on sex. The analysis suggests we should. Missing values for sex are recoded as the mean for each sex.
   
![png](output_9_0.png)

Next, while the particular cabin does not appear to relate to survival, having a cabin assigned does appear to predict survival.  Those assigned a cabin were much more likely to survive than those that were not. Cabin is recoded as a dichotomous variable reflecting assignment or not. Embarked is recoded into into dummy variable to reflect Cherbourg with value of 1 and others with value of 0.

|Assigned a cabin?  | Survived |
| ------------- | ------------- |
|No  | 30.0%  |
| Yes  | 66.7% |

    

There does appear to be a relationship between where the passenger embarked from and survival. Passengers who embarked from Cherbourg (C) were more likely to survive than those embraking from Queenstown (Q) or Southahmpton(S).

|Embarked  | Survived |
| ------------- | ------------- |
|Cherbourg | 55.4%  |
|Queenstown  | 40.0% |
|Southahampton | 33.7% |

Passengers in first were more likely to survive than those in second class. Passengers in second class were more likely to survive than those in third class. 

|Class | Survived |
| ------------- | ------------- |
|First | 70.0%  |
|Second  | 47.2% |
|Third | 24.2% |

   
While sex is expected to relate to survival, it appears that there is also an interaction effect between class, sex, and survival.
   
![png](output_15_0.png)
    
Higher fares, in general, are related to higher survival.

![png](output_16_0.png)
    
Family relationships appear related to survival  Those who survived had fewer siblings and spouses but more parents / children. 
   
![png](output_17_0.png)
      
![png](output_17_1.png)
    
Is there an interaction effect between the variables of SibSp and Parch and survival. The heat map suggests not.
   
![png](output_19_1.png)

I examien the relationship between age and surviving. Age is divided into deciles and plotted against survival. 
    
![png](output_20_0.png)
    
It looks like the very young and very old are more likely to survive but the highest category of age has only one case. We create a dummy variable (child) to represent those aged 0.3 to 8.4 years.

|Age Range | Survived |
| ------------- | ------------- |
|0.3 - 8.4 yrs | 66.7%  |
|8.4 - 16.3 yrs | 41.3% |
|16.3 - 24.3 yrs | 35.6% |
|24.3 - 32.3 yrs | 33.8% |
|32.3 - 40.2 yrs | 44.1% |
|40.2 - 48.2 yrs | 34.3% |
|48.2 - 56.1 yrs | 46.7% |
|56.1 - 64.1 yrs | 37.5% |
|64.1 - 72.0 yrs | 0.0% |
|72.0 - 80.0 yrs | 50.0% |

## Modeling survival on the Titanic

Based on the exploratory data analysis, the model predicting survival on the Titanic will include the following variables:
* Pclass: Original coding as first, second, or third class.
* Sex: Recoded as numeric variable, 1=female, 0=male.
* Sibsp: Original coding as number of siblings and/or spouse.
* Parch: Original coding as number of parents and/or children.
* Cabinassign_code: Cabin recoded as dichotomous variable, 1=cabin assigned, 0=no cabin assigned.
* Cherbourg: Recoding of embarked as dichotomous variables, 1=departed from Cherbourg, 0=departed from another city. 2 missing values for embarked are dropped.
* Child: Age recoded as dichotomous variable 1=ages 0.34-84 years or 0=all others.
* Series of six dichotous variables capturing the interaction between class and sex.

All variables are transformed using StandardScaler to have a mean of 0 and standard deviation of 1 (essentially a z-score).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

subset_df1 = titanic[['Survived', 'Pclass', 'Sex_numeric', 'Parch', 'Fare', 'Cabinassign', 'Cherbourg', 'child', 'ClassSex_female_1', 'ClassSex_female_2', 'ClassSex_female_3', 'ClassSex_male_1', 'ClassSex_male_2', 'ClassSex_male_3']]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(subset_df1)

# Create a new DataFrame with the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=['Survived', 'Pclass', 'Sex_numeric', 'Parch', 'Fare', 'Cabinassign', 'Cherbourg', 'child', 'ClassSex_female_1', 'ClassSex_female_2', 'ClassSex_female_3', 'ClassSex_male_1', 'ClassSex_male_2', 'ClassSex_male_3'])

X = scaled_df.drop('Survived', axis=1)  # Features
y = subset_df1['Survived']  # Target variable

X_train=X
y_train=y

```


```python
titanic_test=pd.read_csv("test.csv")
titanic_test.head(10)

# Recode sex into integer variable
titanic_test['Sex_numeric'] = titanic_test['Sex'].replace({'male': 0, 'female': 1})

# Recode Cabin into Cabinassign
titanic_test['Cabinassign'] = np.where(pd.isna(titanic_test['Cabin']), 0, 1)

# Recode Embarked into Cherbourg
titanic_test['Cherbourg'] = titanic_test['Embarked'].replace({'C': 1, 'Q': 0, 'S': 0})

# Recode Age into child
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test.groupby('Sex')['Age'].transform('mean'))

def recode_variable(Age):
    if Age >= 0.34 and Age <= 8.378:
        return 1
    else:
        return 0

titanic_test['child'] = titanic_test['Age'].apply(recode_variable)

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test.groupby('Pclass')['Fare'].transform('mean'))

titanic_test = titanic_test.dropna(subset=['Cherbourg'])


titanic_test['ClassSex'] =  titanic_test['Sex'].astype(str) + '_' + titanic_test['Pclass'].astype(str)
dummies = pd.get_dummies(titanic_test['ClassSex'], prefix='ClassSex')
titanic_test = pd.concat([titanic_test, dummies], axis=1)


subset_test = titanic_test[['Pclass', 'Sex_numeric', 'Parch', 'Fare', 'Cabinassign', 'Cherbourg', 'child', 'ClassSex_female_1', 'ClassSex_female_2', 'ClassSex_female_3', 'ClassSex_male_1', 'ClassSex_male_2', 'ClassSex_male_3']]

# Standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(subset_test)
X_test = pd.DataFrame(standardized_data, columns=['Pclass', 'Sex_numeric', 'Parch', 'Fare', 'Cabinassign', 'Cherbourg', 'child', 'ClassSex_female_1', 'ClassSex_female_2', 'ClassSex_female_3', 'ClassSex_male_1', 'ClassSex_male_2', 'ClassSex_male_3'])

X_test

missing_values=X_test.isnull().sum()
print(missing_values)

```

    Pclass               0
    Sex_numeric          0
    Parch                0
    Fare                 0
    Cabinassign          0
    Cherbourg            0
    child                0
    ClassSex_female_1    0
    ClassSex_female_2    0
    ClassSex_female_3    0
    ClassSex_male_1      0
    ClassSex_male_2      0
    ClassSex_male_3      0
    dtype: int64
    


```python
# Running the logistic model without tuning parameters

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Ensure X_test is properly defined
y_pred = model.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

#Kaggle competition score=Score: .76794
```

    Submission file created successfully!
    


```python
#Running the logistic model with tuning parameters

from sklearn.model_selection import GridSearchCV

model = LogisticRegression()

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'lbfgs']}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy') 

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = LogisticRegression(**best_params)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

#Kaggle score=.78229

```

    Submission file created successfully!
    

    C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\model_selection\_validation.py:547: FitFailedWarning: 
    50 fits failed out of a total of 200.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    50 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1172, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py", line 67, in _check_solver
        raise ValueError(
    ValueError: Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\sklearn\model_selection\_search.py:1051: UserWarning: One or more of the test scores are non-finite: [0.78624872        nan 0.79751021 0.7941522  0.79187947        nan
     0.79190501 0.79752298 0.79078141        nan 0.78967058 0.78967058
     0.78854699        nan 0.78854699 0.78854699 0.78854699        nan
     0.78854699 0.78854699]
      warnings.warn(
    


```python
#Using Random Forest Classifier without tuning paramters

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)  # Adjust 'n_estimators' (number of trees) as needed
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

#Kaggle competition score=0.76076

```

    Submission file created successfully!
    


```python
# Using Random Forest Classifier with tuning parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np

param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                              n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)

y_pred = rf_random.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

#Kaggle competition score=0.76794
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    

    C:\Users\fh1808mi\AppData\Local\anaconda3\Lib\site-packages\joblib\externals\loky\process_executor.py:752: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      warnings.warn(
    

    Submission file created successfully!
    


```python
# Create KNN classifier (e.g., for k=10 neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
#Kaggle competition score=0.77033 for k=3 nearest neighbors
#Kaggle competition score=0.77511 for k=5 nearest neighbors
#Kaggle competition score=0.76076 for k=10 nearest neighbors
```

    Submission file created successfully!
    


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define parameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.1, 1]
}

# Create SVM model
svm = SVC()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')

# Fit the model with training data
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)

# Get the best model
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
#Kaggle competition score=0.75358

```

    Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
    Submission file created successfully!
    


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create SVM model
svm_model = SVC(kernel='linear', C=1.0) 

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Create submission file (fixing PassengerId reference)
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

#Kaggle competition score=0.77272
```

    Submission file created successfully!
    
