# import required libraries for program
from fileinput import filename
from statistics import mode
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from functions.linear_reg import Ml
import pandas as pd

# from sklearn.datasets import load_files
# import pickle
# import os
# import math

# initialise the application
app = FastAPI()

# initialize model_complete as False and confirmation page variables
model_complete = False
ml_model = ""
in_file = ""
out_file = ""
test_size = 0
model_target = ""
score = 0
score2 = 0

# global variables for prediction
# also uses model_complete
pred_in = ""
pred_out = ""
model_in = ""

# mount the static directory
app.mount('/static', StaticFiles(directory='./static'), name='static')
# mount the templates directory
templates = Jinja2Templates(directory='./templates')

# Home page
@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

# Tutorial page
@app.get('/tutorial')
def tutorial(request: Request):
    return templates.TemplateResponse('tutorial.html', {'request': request})

# Training page
@app.get('/train')
def train(request: Request):
    return templates.TemplateResponse('train.html', {'request': request})

@app.post('/train')
async def train(request: Request, background_tasks : BackgroundTasks, model: str = Form("model"), filename:str = Form("filename"), 
                testSize:float = Form("testSize"), target: str = Form("target")):
    global ml_model
    global model_complete
    global in_file 
    global test_size
    global model_target

    model_target = target
    ml_model = model
    in_file = filename
    test_size = testSize

    print("Filename: ", in_file)
    print(ml_model)
    
    load = Ml(filename=filename)
    df = load.read_csv()

    if not df.empty:
        ml = Ml(dataframe=df, target_column=target)
        ml.load_x_y()

        model_complete = False
        
        if not ml.x.empty and not ml.y.empty:
            background_tasks.add_task(ml.training)
            return templates.TemplateResponse('train-pending.html', {'request': request, 'to_display':[model_target, ml_model, in_file, test_size]})
        else:
            print("Dataset doesn't have the target feature.")

    else:
        print("Error Reading file. Please check the name of the file.")

@app.get('/training-complete')
def training_complete(request: Request):
    global model_complete
    global ml_model
    global in_file
    global out_file
    global test_size
    global model_target
    global score
    global score2


    if model_complete is True:
        return templates.TemplateResponse('training-complete.html', {'request': request, 'to_display':[ml_model, score, score2, test_size, model_target, out_file]})

# Predict page
@app.get('/predict')
def predict(request: Request):
    return templates.TemplateResponse('predict.html', {'request': request})

@app.post('/predict')
async def predict(request: Request, background_tasks : BackgroundTasks, model_path : str = Form("model_path"), data_path : str = Form("data_path")):
    
    #Need to add error handling

    global model_complete
    global pred_in 
    global model_in

    pred_in = data_path
    model_in = model_path

    model_complete = False

    pred_load = Ml(filename=data_path)
    pred_df = pred_load.read_csv()

    predicting = Ml(pred_file_name=model_path, prediction_values=pred_df)
    background_tasks.add_task(predicting.prediction)

    return templates.TemplateResponse('predict-pending.html', {'request': request, 'to_display':[pred_in, model_in]})

@app.get('/predict-complete')
def predict_complete(request: Request):
    global model_complete

    if model_complete is True:
        return templates.TemplateResponse('predict-complete.html', {'request': request, 'to_display': [pred_in, pred_out, model_in]})

if __name__ == '__main__':
    # runs the uvicorn web server for the application to run 
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)


@app.get("/status_check")
def status(request: Request):
    global model_complete
    return model_complete # returns True or False