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
model = ""
in_file = ""
out_file = ""
train_size = 0
target = ""

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
async def train(request: Request, model: str, filename:str, train_size:float, target: str,background_tasks : BackgroundTasks):
    global model_complete
    # print("1",model_complete)
    print("Filename: ", filename)
    print("Train_size Type: ", type(train_size))
    load = Ml(filename=filename)
    df = load.read_csv()
    # print(df)

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
def training_complete(request: Request, model: str, in_file:str,out_file:str,train_size:float,target:str):
    return templates.TemplateResponse('training-complete.html', {'request': request})

# Predict page
@app.get('/predict')
def predict(request: Request):
    return templates.TemplateResponse('predict.html', {'request': request})


if __name__ == '__main__':
    # runs the uvicorn web server for the application to run 
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)