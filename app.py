# import required libraries for program
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# import pandas as pd
# from sklearn.datasets import load_files
# import pickle
# import os
# import math

# initialise the application
app = FastAPI()

# mount the static directory
app.mount('/static', StaticFiles(directory='./static'), name='static')
# mount the templates directory
templates = Jinja2Templates(directory='./templates')

# Home page
@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

# Tutorial page
@app.get('/tutorial')
async def tutorial(request: Request):
    return templates.TemplateResponse('tutorial.html', {'request': request})

# Training page
@app.get('/train')
async def train(request: Request):
    return templates.TemplateResponse('train.html', {'request': request})

# Predict page
@app.get('/predict')
async def predict(request: Request):
    return templates.TemplateResponse('predict.html', {'request': request})


if __name__ == '__main__':
    # runs the uvicorn web server for the application to run 
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)