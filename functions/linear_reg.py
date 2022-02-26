import pandas as pd


# MAIN CLASS FOR LINEAR REGRESSION
class Ml:
    # Intitialising variables to be used in the class names are pretty self explanatory
    def __init__(self, dataframe=pd.DataFrame(), target_column="", pred_file_name="", prediction_values=pd.DataFrame(),
                 test_size=0.2, filename=""):
        self.x = None
        self.y = None
        self.df = dataframe
        self.tc = target_column
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.regressor = None
        self.prediction_filename = pred_file_name
        self.prediction_values = prediction_values
        self.test_size = test_size
        self.fn = filename

    # Read and returns Data frame or 0(In case the loading of the csv Fails)
    def read_csv(self):
        try:
            df = pd.read_csv(self.fn)
            return df
        except Exception as e:
            print(e)
            return None

    # Loading x and y x being the features and y being the target values
    def load_x_y(self):
        df_copy = self.df
        try:
            self.x = df_copy.drop(self.tc, axis=1)
            self.y = self.df[self.tc]
        except Exception as e:
            self.x = 0
            self.y = 0
            print(e)

    # splitting the database into training and testing df
    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size
                                                                                , random_state=0)

    # Training the model pretty straightforward
    async def training(self):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pickle

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size
                                                                                , random_state=0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.x_train, self.y_train)

        filename = 'Trained.pkl'
        pickle.dump(self.regressor, open(filename, 'wb'))

# Predicting from saved model uploaded as pkl or whatever it might take else give out an error statement to inform user
    def prediction(self):
        import pickle
        try:
            file_name = self.prediction_filename
            model_reloaded = pickle.load(open(file_name, 'rb'))
            model_reloaded.predict(self.prediction_values)
            print(predicting_values)
        except Exception as e:
            print(e)


# TEST CODE WITHOUT ANY VALUES PLEASE INSERT THE VARIABLES TO MAKE SURE IT RUNS WITHOUT ERRORS
if __name__ == '__main__':
    # Loading and Training Model and Saving it
    filename = ""  # Load the user's csv file name
    target = "Target Name"  # The target name for the prediction From User's Input or passing it back
    load = Ml(filename=filename)
    df = load.read_csv()
    if df == None:
        print("Error Reading file. Please check the name of the file.")
    else:
        ml = Ml(dataframe=df, target_column=target)
        ml.load_x_y()
        if ml.x != 0 and ml.y != 0:
            ml.training()
        else:
            print("Dataset doesn't have the target feature.")

    # PREDCTION PART

    model_name = ""  # User Uploaded file
    predicting_values = pd.DataFrame()  # User predicting set of values
    predicting = Ml(pred_file_name=model_name, prediction_values=predicting_values)
    predicting.prediction()
