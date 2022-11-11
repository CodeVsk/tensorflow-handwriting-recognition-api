from fastapi import FastAPI
from pydantic import BaseModel
import tensor

app = FastAPI()

class CommandsResult(BaseModel):
    returnCommands: object
    key: str

@app.get("/")
async def root():
    tf = tensor.HandwritingRecognition()
    tf.load_model();
    output = tf.predict_image()

    return {"message": output}