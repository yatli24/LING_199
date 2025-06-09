'''Speech audio feature extraction
'''

# this dictionary defines real and fake audio samples by gender

gender_dict = {
    'male': {
        'real': ['linus', 'musk', 'ryan'],
        'fake': ['linus-to-musk', 'taylor-to-linus', 'linus-to-ryan']
    },
    'female':
    {
        'real': ['taylor', 'margot'],
        'fake': ['taylor-to-margot', 'linus-to-taylor', 'linus-to-margot']
    }
}

real_folder_path = '/content/drive/My Drive/LING_199/REAL_audio'
fake_folder_path = '/content/drive/My Drive/LING_199/FAKE_audio'

def extract_features(real_folder_path, fake_folder_path, gender_dict, gender='both'):
    """
    Extracts audio features from real and fake audio files based on gender specification.

    Parameters:
        real_folder_path (str): path to real audio folders.
        fake_folder_path (str): path to fake audio files.
        gender_dict (dict): Dictionary defining real and fake audio file names by gender.
        gender (str): 'male', 'female', or 'both'.

    Returns:
        DataFrame: Extracted features with labels (0 for real, 1 for fake).

    Function Outline:
    1. Subset audio files based on gender
    2. Open each audio file
        a. Split audio into 1 second frames
        b. Extract features from each frame
        c. Append features to a list
    3. Convert list to dataframe
    """

    # define helper function for processing each file
    def process_file(file_path, label):
        """Loads a file, extracts features from that file, and generates a label
           given a input for label
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

        samples_per_second = sr
        frames = [y[i:i + samples_per_second] for i in range(0, len(y), samples_per_second)]
        features = []

        for frame in frames:
            if len(frame) < samples_per_second:
                continue
            # extract relevant features
            chromagram = librosa.feature.chroma_stft(y=frame, sr=sr).mean()
            rms = librosa.feature.rms(y=frame).mean()
            spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=sr).mean()
            rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(frame).mean()
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20).mean(axis=1)

            frame_features = {
                'chroma_stft': chromagram,
                'rms': rms,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'rolloff': rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                **{f'mfcc{i+1}': mfcc[i] for i in range(20)},
                'LABEL': label
            }
            features.append(frame_features)

        return features

    # Select files based on gender
    if gender.lower() == 'male':
        real_files = gender_dict['male']['real']
        fake_files = gender_dict['male']['fake']
        real_path = os.path.join(real_folder_path, 'MALE')
    elif gender.lower() == 'female':
        real_files = gender_dict['female']['real']
        fake_files = gender_dict['female']['fake']
        real_path = os.path.join(real_folder_path, 'FEMALE')
    else:
        print('Must input male or female')
        return

    real_files = [f + '-original.wav' for f in real_files]
    fake_files = [f + '.wav' for f in fake_files]

    # Feature extraction

    # initiate a list of features
    # each element is a dictionary of features for one second of audio
    all_features = []

    # run process file for all real files, append features to list
    for file in real_files:
        path = os.path.join(real_path, file)
        print(f"Processing real: {path}")
        all_features.extend(process_file(path, label=0))

    # run process file for all fake files, append features to list
    for file in fake_files:
        path = os.path.join(fake_folder_path, file)
        print(f"Processing fake: {path}")
        all_features.extend(process_file(path, label=1))

    # convert list of features to dataframe, return
    return pd.DataFrame(all_features)

'''Usage
data/audio/fake/MALE and data/audio/fake/FEMALE must exist
data/audio/real/MALE and data/audio/real/FEMALE must exist

female_data = extract_features('data/audio/real', 'data/audio/fake', gender_dict, 'female')
male_data = extract_features('data/audio/real', 'data/audio/fake', gender_dict, 'male')
balanced_data = pd.concat([female_data, male_data])
'''
