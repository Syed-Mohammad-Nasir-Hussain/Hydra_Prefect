import pandas as pd
from config import read_params
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
config=read_params()
def data_load():
    data_path=config["data_load"]["data_pull"]
    data=pd.read_csv(data_path)
    return data
def preprocess(data):
    le = preprocessing.LabelEncoder()
    for i in config["data_preprocessing"]['encoding_columns']:
        data[i]=le.fit_transform(data[i])
    for i in config["data_preprocessing"]["values_change"]:
        data[i]=data[i].replace('5more',5)
        data[i]=data[i].replace('more',5)
    features=config["data_preprocessing"]['input']
    x=data[features]
    output=config["base"]["target_col"]
    y=data [output]
    return x,y
def spliting(x,y):
    random_state=config["base"]["random_state"]
    test_size=config["data_preprocessing"]['test_size']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=random_state,test_size=test_size,stratify=y)
    return xtrain,xtest,ytrain,ytest
def model_build(xtrain,xtest,ytrain,ytest):
    n_estimators=config["estimator"]["RF"]['params']['n_estimators']
    max_depth=config["estimator"]["RF"]['params']['max_depth']
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    report=classification_report(ytest,ypred,output_dict=True)
    report=pd.DataFrame(report).transpose()
    scores_file=config['data_preprocessing']['report']
    report.to_csv(scores_file,mode='a+')
data=data_load()
x,y=preprocess(data)
xtrain,xtest,ytrain,ytest=spliting(x,y)
model_build(xtrain,xtest,ytrain,ytest)