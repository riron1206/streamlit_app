"""
Web カメラで得た画像を Streamlit 上で表示して、学習済みの MobileNetV2で分類
https://qiita.com/SatoshiTerasaki/items/f1724d68deecdc14103f

Usage:
    $ poetry run streamlit run ./web_cam_streamlit_tf_class.py
"""
import cv2  # opencv-python==4.2.0.34
import streamlit as st  # streamlit==0.61.0
import tensorflow as tf  # tensorflow==2.2.0
from tensorflow import keras


def get_model():
    model = keras.applications.MobileNetV2(include_top=True, weights="imagenet")
    model.trainable = False
    return model


def get_decoder():
    decode_predictions = keras.applications.mobilenet_v2.decode_predictions
    return decode_predictions


def get_preprocessor():
    def func(image):
        image = tf.cast(image, tf.float32)

        image = tf.image.resize(image, (224, 224))
        image = keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image

    return func


class Classifier:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.model = get_model()
        self.decode_predictions = get_decoder()
        self.preprocessor = get_preprocessor()

    def predict(self, image):
        image = self.preprocessor(image)
        probs = self.model.predict(image)
        result = self.decode_predictions(probs, top=self.top_k)
        return result


def main():
    st.markdown("# Image Classification app using Streamlit")
    st.markdown("model = MobileNetV2")
    device = user_input = st.text_input("input your video/camera device", "0")
    if device.isnumeric():
        device = int(device)
    cap = cv2.VideoCapture(device)
    classifier = Classifier(top_k=5)
    st.markdown("## Image Classification Result top 5 !!!")
    label_names_st = st.empty()
    scores_st = st.empty()
    image_loc = st.empty()

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = classifier.predict(frame)
        labels = []
        scores = []
        for (_, label, prob) in result[0]:
            labels.append(f"{label: <16}")
            s = f"{100*prob:.2f}[%]"
            scores.append(f"{s: <16}")
        label_names_st.text(",".join(labels))
        scores_st.text(",".join(scores))
        image_loc.image(frame)
        if cv2.waitKey() & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()
