## Project Overview

This project uses the **Titanic dataset** to analyze passenger information and build a machine learning model to predict whether a passenger survived the Titanic disaster.

The project follows the basic machine learning process:

**Data Collection → Data Cleaning → Data Visualization → Feature Engineering → Model Training → Model Evaluation**

The main goal was to understand the dataset, find useful patterns, and create a model with good testing accuracy.

## Dataset

The Titanic dataset contains information about passengers, including:

* **Survived** – whether the passenger survived (`0 = No`, `1 = Yes`)
* **Pclass** – passenger class
* **Sex** – passenger gender
* **Age** – passenger age
* **SibSp** – number of siblings or spouses aboard
* **Parch** – number of parents or children aboard
* **Fare** – ticket fare
* **Embarked** – port where the passenger boarded

The `Survived` column was used as the target variable for the machine learning model.

## Data Cleaning and Preprocessing

Before training the model, the dataset was cleaned to handle missing values.

Missing values in the **Age** column were filled using the mean age. Missing values in categorical columns such as **Embarked** were filled using the most common value (mode).

The **Deck** column was removed because it contained too many missing values.

Categorical variables such as `sex` and `embarked` were converted into numerical columns using **one-hot encoding**.

```python
pd.get_dummies(X, columns=["sex", "embarked"], drop_first=True)
```

Using `drop_first=True` removes one category from each group so that the model does not receive unnecessary duplicate information.

## Feature Engineering

Additional features were created to improve the model.

### Family Size

A `family_size` column was created using:

```python
family_size = sibsp + parch + 1
```

This represents the total number of people in a passenger's immediate family traveling with them.

### Fare Transformation

A `fare_log` feature was also created:

```python
fare_log = np.log1p(fare)
```

The log transformation helps reduce the effect of very large fare values and makes the data easier for the model to work with.

## Data Visualization

Several charts were created to understand the Titanic dataset.

The visualizations examined:

* Overall survival
* Survival by gender
* Age distribution
* Survival by passenger class
* Fare distribution
* Embarkation ports
* Family size
* Relationships between passenger features

The charts showed important patterns in the dataset. For example, survival was different between males and females, and passenger class was also related to survival.

## Machine Learning Model

A **Decision Tree Classifier** was used for the prediction.

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(X_train, y_train)
```

The dataset was divided into training and testing data. The training data was used to teach the model, while the testing data was used to check how well the model performed on unseen data.

## Model Accuracy and Tree Depth

Different Decision Tree depths were tested to find a good balance between accuracy and overfitting.

**Depth** refers to the number of levels in the Decision Tree.

A tree with a small depth is simpler, while a tree with a large depth is more complex. If the depth becomes too large, the model can **overfit**, meaning it learns the training data too closely and may perform worse on new data.

The final model achieved approximately **85% testing accuracy**, which was the target for this project.

## Results

The model was evaluated using accuracy:

```python
from sklearn.metrics import accuracy_score

y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

The final results showed that the Decision Tree could predict passenger survival with approximately **85% accuracy** on the testing data.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## Conclusion

This project provided practical experience with the complete machine learning workflow. I learned how to clean missing data, create visualizations, transform features, encode categorical variables, split data into training and testing sets, and train a Decision Tree model.

The project also helped me understand **model accuracy, Decision Tree depth, and overfitting**. By testing different tree depths, I was able to find a model that achieved approximately **85% testing accuracy** while maintaining a reasonable level of complexity.

**Project Goal: Titanic Survival Prediction with approximately 85% testing accuracy.**
