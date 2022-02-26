# import required libraries for program
from fileinput import filename
from statistics import mode
from fastapi import FastAPI, Request, BackgroundTasks
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
mLmodel = ""
in_file = ""
out_file = ""
test_size = 0
model_target = ""
score = 0

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
    print("tutorial Page")
    global model_complete
    print("5",model_complete)
    return templates.TemplateResponse('tutorial.html', {'request': request})

# Training page
@app.get('/train')
def train(request: Request):
    return templates.TemplateResponse('train.html', {'request': request})

@app.post('/train')
async def train(request: Request, model: str, filename:str, testSize:float, target: str,background_tasks : BackgroundTasks):
    global mLmodel
    global model_complete
    global in_file 
    global test_size
    global model_target

    model_target = target
    mLmodel = model
    in_file = filename
    test_size = testSize
    print("Filename: ", filename)
    
    load = Ml(filename=filename)
    df = load.read_csv()

    if not df.empty:
        ml = Ml(dataframe=df, target_column=target)
        ml.load_x_y()

        model_complete = False
        print("2", model_complete)
        if not ml.x.empty and not ml.y.empty:
            background_tasks.add_task(ml.training)
            return templates.TemplateResponse('train-pending.html', {'request': request})
        else:
            print("Dataset doesn't have the target feature.")

    else:
        print("Error Reading file. Please check the name of the file.")

@app.post('/training-complete')
def training_complete(request: Request):
    global model_complete
    global in_file 
    global test_size
    global model_target
    
    
    return templates.TemplateResponse('training-complete.html', {'request': request})

# Predict page
@app.get('/predict')
def predict(request: Request):
    return templates.TemplateResponse('predict.html', {'request': request})


if __name__ == '__main__':
    # runs the uvicorn web server for the application to run 
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)


@app.get("/status_check")
def status(request: Request):
    global model_complete
    return model_complete # returns True or False