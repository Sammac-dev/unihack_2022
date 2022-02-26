import pandas as pd


class Ml:
    def __init__(self, dataframe=pd.DataFrame(), target_column="", pred_file_name="", prediction_values=pd.DataFrame()):
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

    def load_x_y(self):
        df_copy = self.df
        try:
            self.x = df_copy.drop(self.tc, axis=1)
            self.y = self.df[self.tc]
        except:
            self.x = 0
            self.y = 0

    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=0)

    def training(self):
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        self.regressor.fit(self.x_train, self.y_train)

    def output(self):
        import pickle
        filename = 'Trained.pkl'
        pickle.dump(self.regressor, open(filename, 'wb'))

    def prediction(self):
        import pickle
        filename = 'Our_Trained_knn_model.sav'
        model_reloaded = pickle.load(open(filename, 'rb'))
        model_reloaded.predict(self.prediction_values)
        print(predicting_values)


if __name__ == '__main__':
    df = pd.DataFrame()  # Load the user's csv
    target = "Target Name"  # The target name for the prediction
    ml = Ml(dataframe=df, target_column=target)
    ml.load_x_y()
    if ml.x != 0 and ml.y != 0:
        ml.train_test_split()
        ml.training()
        ml.output()
    else:
        print("Dataset doesn't have the target feature.")

    model_name = ""  # User Uploaded file
    predicting_values = pd.DataFrame()  # User predicting set of values

    predicting = Ml(pred_file_name=model_name, prediction_values=predicting_values)
    predicting.prediction()
