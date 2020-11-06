import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

character_map = {0: 'kermit_the_frog',
                 1: 'waldorf_and_statler',
                 2: 'pig',
                 3: 'swedish_chef',
                 4: 'none'}


def labelize_data(movie_path, output_path, image_path, step_size=1):
    cap = cv2.VideoCapture(movie_path)
    last_frame_id = 1
    Path(image_path).mkdir(parents=True, exist_ok=True)

    # if there is already a labeled output file, resume labeling
    if os.path.isfile(output_path):
        input_file = open(output_path, 'r')
        lines = input_file.readlines()
        input_file.close()
        last_line = lines[len(lines)-1]
        last_frame_id = int(last_line.split(',')[0])
        output_file = open(output_path, 'a')
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_id + step_size)
        i = last_frame_id + 2*step_size
    else:
        output_file = open(output_path, 'w')
        i = last_frame_id + step_size

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret:
        print('Failed to read frame %d of video %r.', frame_id, movie_path)
        output_file.close()
        exit(1)

    output_line = str(int(frame_id))
    image_filename = image_path + str(int(frame_id))

    f = plt.ion()
    im = plt.imshow(frame)
    plt.draw()
    label_key = input('[k] kermit, [w] waldorf & statler, [p] pig, [s] swedish chef, [n] none: ')

    if 'stop' in label_key.lower():
        output_file.close()
        exit(1)
    if 'n' in label_key.lower():
        output_file.write(str(frame_id) + ', 4\n')
        cv2.imwrite(image_filename + '_4' + '.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if 'k' in label_key.lower():
        output_line = output_line + ', 0'
        image_filename = image_filename + '_0'
    if 'w' in label_key.lower():
        output_line = output_line + ', 1'
        image_filename = image_filename + '_1'
    if 'p' in label_key.lower():
        output_line = output_line + ', 2'
        image_filename = image_filename + '_2'
    if 's' in label_key.lower():
        output_line = output_line + ', 3'
        image_filename = image_filename + '_3'

    if 'n' not in label_key.lower():
        output_file.write(output_line + '\n')
        cv2.imwrite(image_filename + '.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    while True:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                print('Failed to read frame %d of video %r.', frame_id, movie_path)
                break

            output_line = str(int(frame_id))
            image_filename = image_path + str(int(frame_id))
            im.set_data(frame)
            plt.draw()
            label_key = input('[k] kermit, [w] waldorf & statler, [p] pig, [s] swedish chef, [n] none: ')

            if 'stop' in label_key.lower():
                break
            if 'n' in label_key.lower():
                output_file.write(str(frame_id) + ', 4\n')
                cv2.imwrite(image_filename + '_4' + '.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                i = i + step_size
                continue
            if 'k' in label_key.lower():
                output_line = output_line + ', 0'
                image_filename = image_filename + '_0'
            if 'w' in label_key.lower():
                output_line = output_line + ', 1'
                image_filename = image_filename + '_1'
            if 'p' in label_key.lower():
                output_line = output_line + ', 2'
                image_filename = image_filename + '_2'
            if 's' in label_key.lower():
                output_line = output_line + ', 3'
                image_filename = image_filename + '_3'

            output_file.write(output_line + '\n')
            cv2.imwrite(image_filename + '.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i = i + step_size
        except KeyboardInterrupt:
            break

    output_file.close()
    plt.close()


if __name__ == '__main__':
    test_movie_path = '../../videos/Muppets-02-01-01.avi'
    test_ouput_path = '../../ground_truth/labeled/Muppets-02-01-01.txt'
    test_image_path = '../../ground_truth/labeled/'
    step = 12
    labelize_data(movie_path=test_movie_path, output_path=test_ouput_path, image_path=test_image_path, step_size=step)
