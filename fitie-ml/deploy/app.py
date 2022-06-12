from flask import Flask, render_template, Response
from camera import VideoCamera
from tensorflow import keras

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def gen(camera):
    # extract feature
    IMG_SIZE = 224
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    feature = keras.Model(inputs, outputs, name="feature_extractor")
    while True:
        frame = camera.get_frame(feature)
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
