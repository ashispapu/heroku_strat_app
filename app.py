import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch

class Music(BaseModel):
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    speechiness: float
    tempo: float
    valence: float


app = FastAPI()

"""with open("./model.pkl", "rb") as f:
    model = pickle.load(f)"""

model = torch.load('./fashion_nerda_mbert_model_V2.pt')

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/prediction')
def get_music_category(data: Music):
    """received = data.dict()
    acousticness = received['acousticness']
    danceability = received['danceability']
    energy = received['energy']
    instrumentalness = received['instrumentalness']
    liveness = received['liveness']
    speechiness = received['speechiness']
    tempo = received['tempo']
    valence = received['valence']
    pred_name = model.predict([[acousticness, danceability, energy,
                                instrumentalness, liveness, speechiness, tempo, valence]]).tolist()[0] """

    test_text = "what is the price of tcs ?" 
    pred = model.predict_text(test_text.lower())
    return {'prediction': (pred[0][0], pred[1][0])}


@app.get('/predict')
def get_cat(acousticness: float, danceability: float, energy: float, instrumentalness: float, liveness: float, speechiness: float, tempo: float, valence: float):
    pred_name = model.predict([[acousticness, danceability, energy,
            instrumentalness, liveness, speechiness, tempo, valence]]).tolist()[0]
    return {'prediction': pred_name}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
