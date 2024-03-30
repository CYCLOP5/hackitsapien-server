import streamlit as st
import os
import shutil
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from finalpredicted import predict_deepfake

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error encountered while parsing file: {file_path}")
        return None

def classify_audio(example_file_path):
    loaded_model = joblib.load("./svm_model.joblib")
    example_features = extract_features(example_file_path)
    if example_features is not None:
        prediction = loaded_model.predict([example_features])
        class_label = "Real" if prediction[0] == 1 else "Fake"
        return f"{class_label} Audio File"
    else:
        return "Error extracting features from the example file."

def main():
    st.title("Deepfake Checker")
    st.write("This is a web application that allows users to upload audio and video files for classification and deepfake detection, respectively.")
    
    st.header("Audio Classification")
    uploaded_audio_file = st.file_uploader("Upload Audio File", type=["wav"])
    if uploaded_audio_file is not None:
        st.write("Uploaded audio file details:")
        audio_file_details = {"FileName": uploaded_audio_file.name, "FileType": uploaded_audio_file.type, "FileSize": uploaded_audio_file.size}
        st.write(audio_file_details)

        if st.button("Classify Audio"):
            audio_result = classify_audio(uploaded_audio_file)
            st.write(audio_result)

    st.header("Video Deepfake Detection")
    uploaded_video_file = st.file_uploader("Choose a video file", type=["mp4"])
    method_mapping = {"MTCNN": "plain_frames"}

    if uploaded_video_file is not None:
        selected_option = st.selectbox("Select method", list(method_mapping.keys()))
        st.video(uploaded_video_file)

    st.header("Video Deepfake Detection")
    uploaded_video_file = st.file_uploader("Choose a video file", type=["mp4"])
    method_mapping = {"MTCNN": "plain_frames"}

    if uploaded_video_file is not None:
        selected_option = st.selectbox("Select method", list(method_mapping.keys()))
        st.video(uploaded_video_file)

        method = method_mapping[selected_option]

        if st.button("Check Video"):
            with st.spinner("Checking video..."):
                input_video_file_path = "uploaded_video.mp4"
                with open(input_video_file_path, "wb") as f:
                    f.write(uploaded_video_file.getbuffer())
                fake_prob, real_prob, pred = predict_deepfake(input_video_file_path, method)

            if pred is None:
                st.error("Failed to detect DeepFakes in the video.")
            else:
                label = "real" if pred == 0 else "deepfaked"
                probability = real_prob if pred == 0 else fake_prob
                probability = round(probability * 100, 4)

                if pred == 0:
                    st.success(f"The video is {label}, with a probability of: {probability}%")
                    shutil.rmtree("./output")
                else:
                    st.error(f"The video is {label}, with a probability of: {probability}%")
                    shutil.rmtree("./output")

if __name__ == "__main__":
    main()
