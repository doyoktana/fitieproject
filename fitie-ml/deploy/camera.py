from tensorflow import keras
import imageio
import tensorflow as tf
import cv2
import numpy as np
from tensorflow_docs.vis import embed


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self, feature):
        IMG_SIZE = 224
        MAX_SEQ_LENGTH = 100
        NUM_FEATURES = 2048

        # extracting frames
        ret, frame = self.video.read()
        frame2 = frame

        # crop center square
        resize = (IMG_SIZE, IMG_SIZE)
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        frame = frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

        # load video
        frames = []
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)
        frames = np.array(frames)

        # prepare video
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = feature.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        model = tf.keras.models.load_model('model2.h5')
        probabilities = model.predict([frame_features, frame_mask])[0]
        class_vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        v = ['Bench Press', 'Body Weight Squats', 'Handstand Push Up', 'Jump Rope', 'Jumping Jack',
             'Leg Raise', 'Lunges', 'Overhead Press', 'Pull Up', 'Push Up']
        max = 0
        res = ""
        for i in np.argsort(probabilities)[::-1]:
            if(max < probabilities[i]):
                max = probabilities[i]
                res = v[int(class_vocab[i])] + " : " + str(round(probabilities[i]*100, 2)) + "%"
            # print(res)


        # draw the top prediction
        x, y, w, h = 0, 0, 250, 50
        cv2.rectangle(frame2, (x, x), (x + w, y + h), (0, 0, 0), -1)
        cv2.putText(frame2, res, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame2)
        return jpeg.tobytes()
