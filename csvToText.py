import ast
from pathlib import Path

conversion_map = {'k': 0,
                  'w': 1,
                  'p': 2,
                  's': 3,
                  'n': 4}


def csv_to_txt(csv_filename, output_filename, step):
    # read mapping from csv format
    label_mapping = {}
    with open(csv_filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            id = int(line.split("\"")[0].split(',')[0])
            if id == 0:
                id = 1
            elif id == 1:
                id = id + step
            else:
                id = step * id + 1

            label_mapping[id] = ast.literal_eval(line.split('\"')[1])

    # write mapping in txt format
    with open(output_filename, 'w') as f:
        for key, value in label_mapping.items():
            labels = ''.join([''.join(str(conversion_map[k]) + ', ') for k, val in value.items() if val == 1])
            f.write('%d, %s\n' % (key, labels[:-2]))


if __name__ == '__main__':
    in_filename = 'labels.csv'
    out_filename = 'ground_truth/Muppets-02-04-04/Muppets-02-04-04.txt'
    Path('ground_truth/Muppets-02-04-04').mkdir(parents=True, exist_ok=True)
    step = 12
    csv_to_txt(in_filename, out_filename, step)
