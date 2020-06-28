import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class CategoricalImputer(BaseEstimator,TransformerMixin):
    
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables=[variables]
        else:
            self.variables=variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature]=X[feature].fillna('Missing')
        return X

class NumericalImputer(BaseEstimator,TransformerMixin):
        def __init__(self,variables=None):
            if not isinstance(variables,list):
                self.variables=[variables]
            else:
                self.variables=variables
        
        def fit(self,X,y=None):
            self.impute_dict_={}
            for feature in self.variables:
                self.impute_dict_[feature]=X[feature].mode()[0]
            return self

        def transform(self,X):
            X=X.copy()
            for feature in self.variables:
                X[feature].fillna(self.impute_dict_[feature],inplace=True)
            return X

class TemporalVariableEstimator(BaseEstimator,TransformerMixin):

    def __init__(self,variables=None,reference_variable=None):
        if not isinstance(variables,list):
            self.variables=[variables]
        else:
            self.variables=variables
        self.reference_variable=reference_variable

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature]=X[self.reference_variable]-X[feature]
        return X

class RareLabelCategoricalEncoder(BaseEstimator,TransformerMixin):
    
    def __init__(self,tol=0.05,variables=None):
        self.tol=tol
        if not isinstance(variables,list):
            self.variables=[variables]
        else:
            self.variables=variables
    
    def fit(self,X,y=None):
        self.encoder_dict_={}
        for var in self.variables:
            t=pd.Series(X[var].value_counts() / np.float(len(X)))
            self.encoder_dict_[var]=list(t[t>=self.tol].index)
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature]=np.where(X[feature].isin(self.encoder_dict_[feature]),X[feature],'Rare')
        
        return X

class CategoricalEncoder(BaseEstimator,TransformerMixin):

    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables=[variables]
        else:
            self.variables=variables
        
    def fit(self,X,y):
        tmp=pd.concat([X,y],axis=1)
        tmp.columns=list(X.columns)+['target']
        self.encoder_dict_={}
        for var in self.variables:
            t=tmp.groupby([var])['target'].mean().sort_values(ascending=True).index
            self.encoder_dict_[var]={k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature]=X[feature].map(self.encoder_dict_[feature])

        # 
        return X

class LogTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables=[variables]
        else:
            self.variables=variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.copy()
        # if not (X[self.variables]>0).all().all():
        #     vars_=self.variables[(X[self.variables]<=0).any()]
        #     raise errors.InvalidModelInputError(
        #         f"Variables contain zero or negative values, "
        #         f"can't apply log for vars: {vars_}"
        #     )
        for feature in self.variables:
            X[feature]=np.log(X[feature])

        return X

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X