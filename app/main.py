from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import librosa
import io
import tempfile

app = FastAPI()

# Load models
cry_model = load_model("cry_detection_cnn.keras")
child_model = load_model("child_adult_model.h5")

@app.get("/")
def home():
    return {"message": "AI models are running successfully!"}

# ---- AUDIO ENDPOINT ----
@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load audio
    y, sr = librosa.load(tmp_path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=(0, -1))

    result = cry_model.predict(mfcc)
    label = "crying" if result[0][0] > 0.5 else "not crying"
    return {"result": label, "confidence": float(result[0][0])}

# ---- IMAGE ENDPOINT ----
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = child_model.predict(x)
    label = "child" if pred[0][0] > 0.5 else "adult"
    return {"result": label, "confidence": float(pred[0][0])}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)