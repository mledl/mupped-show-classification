import math
import random
import cv2
from pathlib import Path

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


def create_image_dataset_for_character(character_id, data_locations_dict):
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
                         ground_truth_files_base_path + 'kermit/')


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


if __name__ == '__main__':
    ground_truth_txt_files = ['../../ground_truth/Muppets-02-01-01/Muppets-02-01-01.txt',
                              '../../ground_truth/Muppets-02-04-04/Muppets-02-04-04.txt',
                              '../../ground_truth/Muppets-03-04-03/Muppets-03-04-03.txt']

    ground_truth_locations = parse_ground_truth_txt_files(ground_truth_txt_files)
    print_ground_truth_statistics(ground_truth_locations)
    create_image_dataset_for_character(0, ground_truth_locations)
