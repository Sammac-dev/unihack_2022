from flask import Flask, render_template
from flask.helpers import url_for
import pandas as pd
from sklearn.datasets import load_files
import pickle
import os
import math

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    # code for home page here
    # if using page for processing need to use methods=['GET','POST'] in params
    return render_template('index.html')