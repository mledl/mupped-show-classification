import math
import random
import cv2
import glob
import librosa
import os
from pathlib import Path
from audio_extractor import extract_audio_snippets

character_map = {0: 'kermit_the_frog',
                 1: 'waldorf_and_statler',
                 2: 'pig',
                 3: 'swedish_chef',
                 4: 'none'}

file_map = {'Muppets-02-01-01.txt': 1,
            'Muppets-02-04-04.txt': 2,
            'Muppets-03-04-03.txt': 3}

video_base_path = '../../videos/'
ground_truth_files_base_path = '../../ground_truth/'
audio_snippet_path = '../../audio/'
mfcc_feature_file = '../../ground_truth/audio/mfcc.txt'
ground_truth_txt_files = ['../../ground_truth/Muppets-02-01-01/Muppets-02-01-01.txt',
                          '../../ground_truth/Muppets-02-04-04/Muppets-02-04-04.txt',
                          '../../ground_truth/Muppets-03-04-03/Muppets-03-04-03.txt']


def print_ground_truth_statistics(data_locations_dict):
    """
    The aim of this method is to print statistics of the ground truth.
    :param data_locations_dict: dict holding the ground truth location data
    """
    character_location_map = {}
    total_samples = 0
    print('Number of samples per character in ground truth:')
    for i in range(0, len(character_map)):
        no_of_samples = 0
        for key, data_locations in data_locations_dict.items():
            character_location_map[key] = data_locations[i]
            no_of_samples += len(data_locations[i])
        total_samples += no_of_samples
        print('%s: %d' % (character_map[i], no_of_samples))
    print('total_samples: %d' % total_samples)


def extract_ground_truth(character_location_map, rest_location_map, character_id, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    labels_file = open(output_path + 'labels.txt', 'w')
    labels_file.write('txt_file, frame_id, label\n')

    # write images of actual target character
    print('[INFO] Start extracting images for target class: %d' % character_id)
    for key, values in character_location_map.items():
        video_path = video_base_path + key.split('.')[0] + '.avi'
        cap = cv2.VideoCapture(video_path)

        for value in values:
            cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame %d of video %r.' % (value, video_path))
                labels_file.close()
                exit(1)

            filename = '%s/%d_%d_%d.jpg' % (output_path, file_map[key], value, character_id)
            labels_file.write('%d, %d, %d\n' % (file_map[key], value, character_id))
            cv2.imwrite(filename, frame)

    print('[INFO] Start extracting randomly sampled images')
    for key, values in rest_location_map.items():
        for k, vals in values.items():
            video_path = video_base_path + k.split('.')[0] + '.avi'
            cap = cv2.VideoCapture(video_path)

            for val in vals:
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)
                ret, frame = cap.read()
                if not ret:
                    print('Failed to read frame %d of video %r.' % (val, video_path))
                    labels_file.close()
                    exit(1)

                filename = '%s/%d_%d_%d.jpg' % (output_path, file_map[k], val, key)
                labels_file.write('%d, %d, %d\n' % (file_map[k], val, key))
                cv2.imwrite(filename, frame)

    labels_file.close()


def create_image_dataset_for_character(character_id, data_locations_dict, sub_path):
    """
    The aim of this method is to generate a dataset for the specified character that consists of
    50% images labeled with the specified character and 50% randomly sampled of all others
    :param character_id: the id of the character
    :param data_locations_dict: dict holding the ground truth location data
    :return:
    """
    character_location_map = {}
    half_length = 0
    for key, data_locations in data_locations_dict.items():
        character_location_map[key] = data_locations[character_id]
        half_length += len(data_locations[character_id])

    # calculate data distribution over ground truth and per video
    data_distribution_map = {}
    total_samples = 0
    for i in range(0, len(character_map)):
        if i != character_id:
            temp = {}
            for key, data_locations in data_locations_dict.items():
                total_samples += len(data_locations[i])
                temp[key] = len(data_locations[i])
            data_distribution_map[i] = temp

    # calculate absolute rest distribution map
    rest_data_distribution_map = {}
    for key, values in data_distribution_map.items():
        temp = {}
        for k, v in values.items():
            temp[k] = math.ceil((v / total_samples) * half_length)
        rest_data_distribution_map[key] = temp

    # actually do the random sampling
    rest_frameid_map = {}
    random.seed(333)
    for key, values in rest_data_distribution_map.items():
        temp = {}
        for k, v in values.items():
            temp[k] = random.sample(data_locations_dict[k][key], v)
        rest_frameid_map[key] = temp

    extract_ground_truth(character_location_map, rest_frameid_map, character_id,
                         ground_truth_files_base_path + sub_path)


def parse_ground_truth_txt_files(ground_truth_files):
    """
    The aim of this method is to parse the ground truth from corresponding text files.
    :param ground_truth_files: a list of ground truth text file paths
    :return: a dictionary representing the ground truth locations
    """
    parsed_ground_truth = {}
    for filename in ground_truth_files:
        gt = {}
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                str_parts = line.strip().split(', ')
                parts = [int(p) for p in str_parts]
                for part in parts[1:]:
                    try:
                        gt[part].append(parts[0])
                    except KeyError:
                        gt[part] = [parts[0]]

        parsed_ground_truth[filename.split('/')[-1]] = gt

    return parsed_ground_truth


def create_mfcc_audio_dataset(audio_path, frame_length_ms, n_mfcc, output_file):
    # extract counts for snippets with and without given character
    total_no_audios = len(glob.glob(audio_path + '*.wav'))
    print('Total number of audio snippets: %d' % total_no_audios)
    print('Window size: %d ms' % frame_length_ms)
    print('Number of MFCC features: %d' % n_mfcc)
    print('Extracting MFCC features for audio data...')

    # define fft window and sliding window factors based on given frame length
    mfcc_n_fft_factor = frame_length_ms / 1000  # window factor
    mfcc_hop_length_factor = mfcc_n_fft_factor * 0.5  # sliding window factor, note that this must be an int

    # extract MFCC features for all audio files
    mfcc_audio_data = {}
    for audio_file in glob.glob(audio_path + '*.wav'):
        # extract file id and character id
        filename = audio_file.split('/')[-1]
        file_char_id = filename.split('_')[0][-1] + '_' + filename.split('_')[1]

        raw_data, sample_rate = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=n_mfcc,
                                     hop_length=int(mfcc_hop_length_factor * sample_rate),
                                     n_fft=int(mfcc_n_fft_factor * sample_rate)).T

        try:
            mfcc_audio_data[file_char_id].append(mfccs)
        except KeyError:
            mfcc_audio_data[file_char_id] = [mfccs]

    # write calculated MFCCs to file
    print('Write extracted MFCCs to file: %s' % output_file)
    with open(output_file, 'w') as f:
        for key, values in mfcc_audio_data.items():
            file_id = key.split('_')[0]
            char_id = key.split('_')[1]
            for mfcc_array in values:
                for mfcc_values in mfcc_array:
                    list_as_string = ','.join([str(mfcc_values[i]) for i in range(0, mfcc_array.shape[1])])
                    f.write('%s, %s, %s\n' % (file_id, char_id, list_as_string))


def random_sample_mfcc(target_character_id, mfcc_file):
    # read the mfcc features from file
    print('Read MFCC features for random sampling...')
    mfcc_data_all = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
    total_number_of_samples = 0
    no_positive_samples = 0
    with open(mfcc_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.split(',')
            file_id = int(parts[0].strip())
            char_id = int(parts[1].strip())
            mfcc_coeffs = [float(parts[i].strip()) for i in range(2, len(parts))]

            if char_id == target_character_id:
                no_positive_samples += 1

            try:
                mfcc_data_all[char_id][file_id].append(mfcc_coeffs)
            except KeyError:
                mfcc_data_all[char_id][file_id] = [mfcc_coeffs]
            total_number_of_samples += 1

    # exract the number of sample present for target character
    print('Number of samples for target class %d: %d' % (target_character_id, no_positive_samples))

    # calculate data distribution
    print('Create data distribution map...')
    data_distribution_map = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
    no_rest_samples = total_number_of_samples - no_positive_samples
    for char_id, value in mfcc_data_all.items():
        if char_id != target_character_id:
            for file_id, mfccs in value.items():
                data_distribution_map[char_id][file_id] = math.ceil(
                    (len(mfccs) / no_rest_samples) * no_positive_samples)

    # add positive samples to resulting dataset
    dataset = []
    for char_id, value in mfcc_data_all.items():
        if char_id == target_character_id:
            for file_id, mfccs in value.items():
                dataset += [(1, file_id, mfcc) for mfcc in mfccs]

    # randomly sample the negative samples according to data distribution
    random.seed(333)
    for char_id, value in data_distribution_map.items():
        for file_id, k in value.items():
            dataset += [(0, file_id, mfcc) for mfcc in random.sample(mfcc_data_all[char_id][file_id], k)]

    print('Successfully extracted MFCC feature dataset for character: %d' % target_character_id)

    return dataset


def get_waldorf_statler_mfcc_features(frame_length_ms, n_mfcc):
    Path('../../ground_truth/audio/').mkdir(parents=True, exist_ok=True)

    # check if audio snippets have alerady been extracted
    if len(os.listdir('../../audio')) == 0:
        extract_audio_snippets()

    # if mfcc data has not been extracted, call the extraction
    if len(os.listdir('../../ground_truth/audio/')) == 0:
        create_mfcc_audio_dataset(audio_snippet_path, frame_length_ms, n_mfcc, mfcc_feature_file)

    return random_sample_mfcc(1, mfcc_feature_file)


def create_kermit_image_dataset():
    Path('../../ground_truth/kermit/').mkdir(parents=True, exist_ok=True)

    # extract kermit image dataset if not already created
    if len(os.listdir('../../ground_truth/kermit/')) == 0:
        ground_truth_locations = parse_ground_truth_txt_files(ground_truth_txt_files)
        print_ground_truth_statistics(ground_truth_locations)
        create_image_dataset_for_character(0, ground_truth_locations, 'kermit/')
    else:
        print('Kermit image dataset already created.')

def create_pig_image_dataset():
    Path('../../ground_truth/pig/').mkdir(parents=True, exist_ok=True)

    # extract kermit image dataset if not already created
    if len(os.listdir('../../ground_truth/pig/')) == 0:
        ground_truth_locations = parse_ground_truth_txt_files(ground_truth_txt_files)
        print_ground_truth_statistics(ground_truth_locations)
        create_image_dataset_for_character(2, ground_truth_locations, 'pig/')
    else:
        print('Pigs image dataset already created.')

def create_swedish_chef_image_dataset():
    Path('../../ground_truth/swedish_chef/').mkdir(parents=True, exist_ok=True)

    # extract kermit image dataset if not already created
    if len(os.listdir('../../ground_truth/swedish_chef/')) == 0:
        ground_truth_locations = parse_ground_truth_txt_files(ground_truth_txt_files)
        print_ground_truth_statistics(ground_truth_locations)
        create_image_dataset_for_character(3, ground_truth_locations, 'swedish_chef/')
    else:
        print('Swedish Chef image dataset already created.')
