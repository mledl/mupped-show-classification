import cv2
from pathlib import Path


def extract_ground_truth_images(ground_truth_textfile, video_path, image_path):
    cap = cv2.VideoCapture(video_path)
    Path(image_path).mkdir(parents=True, exist_ok=True)

    with open(ground_truth_textfile, 'r') as f:
        for line in f:
            entries = line.split(', ')
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(entries[0]))
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame %d of video %r.', entries[0], movie_path)
                exit(1)

            filename = image_path + '/' + str(entries[0]) + '_' + \
                       '_'.join([str(label).rstrip() for label in entries[1:]]) + '.png'
            cv2.imwrite(filename, frame)


if __name__ == '__main__':
    test_video_path = '../../videos/Muppets-02-01-01.avi'
    test_ground_truth_textfile = '../../ground_truth/Muppets-02-01-01/Muppets-02-01-01.txt'
    test_image_path = '../../ground_truth/labeled'

    extract_ground_truth_images(ground_truth_textfile=test_ground_truth_textfile, video_path=test_video_path,
                                image_path=test_image_path)
