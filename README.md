# rf-prox

Function to calculate the proximity matrix for random forest.

It should work for all sklearn ensemble models with 'apply' method

#### Example Usage
    from proximityMatrix import proximityMatrix
  
#### Import libraries
    import pandas as pd
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier

#### Read in data and preprocess
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    features = df.columns[:4]
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    y = pd.factorize(df['species'])[0]
    
#### Fit Random Forest on all data
    clf = RandomForestClassifier(random_state=0, n_estimators=100, oob_score=True)
    clf.fit(df[features], y)
    
#### Calculate proximity matrix
    print('\nnormalize=True, dist=True')
    print(proximityMatrix(clf, df[features], normalize=True, dist=True))

    print('\nnormalize=False, dist=True')
    print(proximityMatrix(clf, df[features], normalize=False, dist=True))

    print('\nnormalize=True, dist=False')
    print(proximityMatrix(clf, df[features], normalize=True, dist=False))

    print('\nnormalize=False, dist=False')
    print(proximityMatrix(clf, df[features], normalize=False, dist=False))
