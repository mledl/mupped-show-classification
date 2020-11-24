from moviepy.editor import *
import os
from pathlib import Path
from pydub import AudioSegment

character_map = {0: 'kermit_the_frog',
                 1: 'waldorf_and_statler',
                 2: 'pig',
                 3: 'swedish_chef',
                 4: 'none'}


def extract_audio_from_video(video_path, audio_base_path):
    # extract audio from avi video
    print('[INFO] Start extracting wav from avi')
    filename = audio_base_path + os.path.basename(os.path.normpath(video_path)).split('.')[0] + '.wav'
    if not os.path.isfile(filename):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(filename)
        video.close()
        print('[INFO] Finished extracting wav from avi')
    else:
        print('[INFO] Wav File already exists')


def extract_character_screentime(ground_truth_textfile):
    print('[INFO] Start calculating screen times for characters')
    screen_time_dict = {}
    for key in character_map:
        screen_time_dict[key] = screentime_per_class(ground_truth_textfile, key)

    print('[INFO] Finished calculating screen times for characters')
    return screen_time_dict


def screentime_per_class(ground_truth_textfile, class_label):
    with open(ground_truth_textfile) as file:
        lines = file.readlines()

    intervals = []
    in_interval = False
    interval_start_frame_id = -1
    interval_end_frame_id = -1
    for line in lines[1:]:
        splits = line.strip().split(',')
        frame_id = splits[0]
        labels = [int(splits[i]) for i in range(1, len(splits))]

        if class_label in labels:
            if in_interval is False:
                interval_start_frame_id = frame_id
                in_interval = True
        else:
            if in_interval is True:
                interval_end_frame_id = frame_id
                in_interval = False
                intervals.append((interval_start_frame_id, interval_end_frame_id))

    return intervals


def slice_audio_from_video(ground_truth_textfile, audio_path, audio_base_path, video_fps, file_id):
    screen_time_map = extract_character_screentime(ground_truth_textfile)
    audio = AudioSegment.from_wav(audio_path)

    print('[INFO] Start slicing audio files')
    for key, value in screen_time_map.items():
        print('[INFO] Start slicing for label: %d' % key)
        for i, interval in enumerate(value):
            start = (float(interval[0]) / video_fps) * 1000
            end = (float(interval[1]) / video_fps) * 1000
            audio_chunk = audio[start:end]
            audio_chunk.export(audio_base_path + str(file_id) + '_' + str(key) + '_' + str(i) + '.wav', format='wav')
        print('[INFO] Finished slicing for label: %d' % key)


if __name__ == '__main__':
    video_paths = ['../../videos/Muppets-02-01-01.avi', '../../videos/Muppets-02-04-04.avi',
                   '../../videos/Muppets-03-04-03.avi']
    test_audio_base_path = '../../audio/'
    audio_paths = [test_audio_base_path + 'Muppets-02-01-01.wav', test_audio_base_path + 'Muppets-02-04-04.wav',
                   test_audio_base_path + 'Muppets-03-04-03.wav']
    ground_truth_textfiles = ['../../ground_truth/Muppets-02-01-01/Muppets-02-01-01.txt',
                              '../../ground_truth/Muppets-02-04-04/Muppets-02-04-04.txt',
                              '../../ground_truth/Muppets-03-04-03/Muppets-03-04-03.txt']
    fps = 25

    Path(test_audio_base_path).mkdir(parents=True, exist_ok=True)

    for path in video_paths:
        extract_audio_from_video(video_path=path, audio_base_path=test_audio_base_path)

    for i in range(0, len(audio_paths)):
        slice_audio_from_video(ground_truth_textfile=ground_truth_textfiles[i], audio_path=audio_paths[i],
                               audio_base_path=test_audio_base_path, video_fps=fps, file_id=i + 1)
        os.remove(audio_paths[i])
