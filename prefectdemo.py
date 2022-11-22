import pandas as pd
import hydra
import prefect
from prefect import task,flow
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from hydra.utils import get_original_cwd,to_absolute_path
from helper import load_config
from omegaconf import DictConfig
cfg=load_config()
import logging
@task()
def data_load(cfg):
    logging.basicConfig(filename=to_absolute_path(cfg.log_path.path),level=logging.INFO,
                format='%(levelname)s:%(asctime)s:%(message)s')
    logging.info("data_load Started")
    data_path=to_absolute_path(cfg.data_load.data_pull)
    data=pd.read_csv(data_path)
    return data
@task()
def preprocess(cfg,data):
    logging.info("Preprocessing Started")
    le = preprocessing.LabelEncoder()
    logging.info("Encoding Started")
    for i in cfg.data_preprocessing.encoding_columns:
        data[i]=le.fit_transform(data[i])
    logging.info("Encoding Ended")
    for i in cfg.data_preprocessing.values_change:
        data[i]=data[i].replace('5more',5)
        data[i]=data[i].replace('more',5)
    logging.info("Splitting Started")
    features=cfg.data_preprocessing.input
    x=data[features]
    output=cfg.base.target_col
    y=data [output]
    return x,y
@task()
def spliting(cfg,x,y):
    random_state=cfg.base.random_state
    test_size=cfg.data_preprocessing.test_size
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=random_state,test_size=test_size,stratify=y)
    logging.info("Splitting Ended")
    return xtrain,xtest,ytrain,ytest
@task()
def model_build(cfg,xtrain,xtest,ytrain,ytest):
    logging.info("Model Build Started")
    n_estimators=cfg.estimator.RF.params.n_estimators
    max_depth=cfg.estimator.RF.params.max_depth
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    model.fit(xtrain,ytrain)
    logging.info("Model Build Ended")
    ypred=model.predict(xtest)
    report=classification_report(ytest,ypred,output_dict=True)
    report=pd.DataFrame(report).transpose()
    scores_file=cfg.data_preprocessing.report
    logging.info("Report Saved")
    report.to_csv(scores_file,mode='a+')
# @hydra.main(config_path="../conf",config_name="config.yaml")
# if we havent created helper file we should have used above decorator 
#def model_run(cfg:DictConfig) 
#we should have used DictConfig
# if we want to utilise multiple yaml files wihout creating functions we use decorator before
# the function and specify your yaml file in config path and config name
# we can pass the config parameters through command line which will override the values.
# Ex: python src\hydrademo.py ++base.random_state=10
@flow(name='Car_Acceptance')
def model_run():
    data=data_load(cfg)
    x,y=preprocess(cfg,data)
    xtrain,xtest,ytrain,ytest=spliting(cfg,x,y)
    model_build(cfg,xtrain,xtest,ytrain,ytest)
model_run()