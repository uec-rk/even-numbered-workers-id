# Instruction for workers with even-numbered Workers ID

## Requirements

We require:

+ Python (>3.6)

## Settings

1. Create a virtual environment.

    `> python -m venv venv`

1. Activate the virtual environment.

    + Windows

        `(venv)> venv\Scripts\activate`

    + Mac, Linux

        `(venv)$ source/bin/activate`

1. Install libraries

    `(venv)> pip install git+https://github.com/uec-rk/malss`

1. Download training data and test data

    + [X_train.csv](https://drive.google.com/file/d/14sr7seytrlncsxY0Jtqf0SbXGQ7KnELt/view?usp=sharing)

    + [y_train.csv](https://drive.google.com/file/d/1pWW-bAgyaIPkk8m_3NntnN8Z-T0UBY0g/view?usp=sharing)

    + [X_test.csv](https://drive.google.com/file/d/1kYqLo-ekUmS3V0ph3-eEJBztcm1LuL-f/view?usp=sharing)

    + [y_test.csv](https://drive.google.com/file/d/1GTJqUZcqXZvpdFe6F79olLfDHydLi-hP/view?usp=sharing)

## Data analysis experiment  

You will be asked to perform a simple data analysis task.
The steps of the analysis are as follows:

1. Make a model

    The task is a **<font color="orange">regression task</font>**.  
    You will make a model that predict _Target_ variable in **<font color="orange">y_train.csv</font>** using **<font color="orange">X_train.csv</font>**.  
    Variables in the X_train.csv are _Cat1_, _Cat2_, _Num1_, ..., _Num19_.  
    Note that **<font color="orange">_Cat1_ and _Cat2_ are categorical variables</font>**.

    You need to make a model using a specific python library, **MALSS**.
    You can see the usage of the MALSS in the _usage section_ below.

    Try to make a better performance model using the MALSS.  
    The estimated maximum working time is 1 hour.

1. Evaluate the performance of the model

    Evaluate the performance of the model using **<font color="orange">X_test.csv and y_test.csv</font>**.  
    The results are saved as _result.txt_ in the current directory.  
    **Copy the text in the result.txt and paste it to the form in the machine learning survey.**

    ```
    # Assume that the model has already been made as model.

    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    model.predict(X_test, y_test)
    ```

## Usage of MALSS

1. Make a model

    ```
    import pandas as pd
    from malss import MALSS

    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    model = MALSS(task='regression')
    model.fit(X_train.csv, y_train.csv, 'results)
    # An analysis report are made in the results directory.
    # Please carefully read the report.
    ```

1. Tune hyper-parameters

    ```
    import pandas as pd
    from sklearn.svm import SVR
    from malss import MALSS

    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    model = MALSS(task='regression')
    model.fit(X_train.csv, y_train.csv, algorithm_selection_only=True)

    # Check algorithms
    print(model.get_algorithms())

    # Remove the algorithm that you want to tune hyper-parameters
    model.remove_algorithm(0)  # Set the index of the algorithm

    # Add SVR (Support Vector Regressor) with hyper-parameters
    model.add_algorithm(
        SVR(kernel='rbf'),  # Instance of the algorithm
        [{'C': [300, 1000, 3000, 10000],  # Hyper-parameters
          'gamma': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}],
        'Support Vector Machine (RBF Kernel)')  # Algorithm name for analysis report
    model.fit(X_train, y_train, dname='default')
    model.predict(X_test, y_test)

    model.fit(X_train.csv, y_train.csv, 'results2')
    ```

    Algorithms:

    + [Support Vector Regressor (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

    + [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest%20regressor#sklearn.ensemble.RandomForestRegressor)

    + [Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html?highlight=ridge%20regression#sklearn.linear_model.ridge_regression)

    + [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree#sklearn.tree.DecisionTreeRegressor)

1. Select features

    ```
    import pandas as pd
    from malss.malss import MALSS

    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    model = MALSS(task='regression')
    model.fit(X_train.csv, y_train.csv, 'results')
    # Check the analysis report in the results directory.

    model.select_features()
    model.fit(X_train.csv, y_train.csv, 'results3')
    # You can set the original data after feature selection
    # (You do not need to select features by yourself.)
    ```