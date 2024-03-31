import streamlit as st
import os
import shutil
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from finalpredicted import predict_deepfake
import threading
from queue import Queue

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error encountered while parsing file: {file_path}")
        return None

def classify_audio(example_file_path):
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(current_dir)

    loaded_model = joblib.load("svm_model.joblib")

    example_features = extract_features(example_file_path)
    if example_features is not None:
        prediction = loaded_model.predict([example_features])
        class_label = "Real" if prediction[0] == 1 else "Fake"
        return f"{class_label} Audio File"
    else:
        raise Exception("Unknown method")

    if verbose:
        print(f'Detecting DeepFakes using method: {df_method}')

    current_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(current_dir)
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    if verbose:
        print(f'Loading model weights {model_path}')
    check_point_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(check_point_dict['model_state_dict'])

    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                             pin_memory=True)
    if len(test_loader) == 0:
        print('Cannot extract images. Dataloaders empty')
        return None, None, None
    probabilities = []
    all_filenames = []
    all_predicted_labels = []
    with torch.no_grad():
        for batch_id, samples in enumerate(test_loader):
            frames = samples[0].to(device)
            output = model(frames)
            predicted = get_predictions(output).to('cpu').detach().numpy()
            class_probability = get_probability(output).to('cpu').detach().numpy()
            if len(predicted) > 1:
                all_predicted_labels.extend(predicted.squeeze())
                probabilities.extend(class_probability.squeeze())
                all_filenames.extend(samples[1])
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
