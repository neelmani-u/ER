import os.path
import pickle
# import pandas as pd
# from sklearn.metrics import f1_score
from data_preprocessing import extract_feature
from train import y_test, y_pred, model


# f1_score(y_test, y_pred, average=None)
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df.head(20)

filename = 'modelForPrediction1.sav'
# check if model exists or not
if not os.path.exists(filename):
    # Writing different model files to file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# loading the model file from the storage
loaded_model = pickle.load(open(filename, 'rb'))


def analyze_output(audio_file_path, mfcc, chroma, mel):
    feature = extract_feature(audio_file_path, mfcc=mfcc, chroma=chroma, mel=mel)
    feature = feature.reshape(1, -1)
    prediction = loaded_model.predict(feature)
    print(prediction)


analyze_output("./training-data/Actor_01/03-01-01-01-01-01-01.wav", mfcc=True, chroma=True, mel=True)
