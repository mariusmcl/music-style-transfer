import csv
import os
from pathlib import Path
from shutil import copy

import pandas
from intervaltree import IntervalTree


def process_labels(root, path):
    trees = dict()
    for item in os.listdir(os.path.join(root, path)):
        if not item.endswith('.csv'): continue
        uid = int(item[:-4])
        tree = IntervalTree()
        with open(item, 'rb') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                instrument = int(label['instrument'])
                note = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat = float(label['end_beat'])
                note_value = label['note_value']
                tree[start_time:end_time] = (instrument, note, start_beat, end_beat, note_value)
        trees[uid] = tree
    return trees


def split_domains(domains, in_dir, out_dir, meta_data):
    train_dir = in_dir / 'train_data'
    test_dir = in_dir / 'test_data'

    for (ensemble, composer) in domains:
        fid_list = meta_data[(meta_data["composer"] == composer) & (meta_data["ensemble"] == ensemble)].id.tolist()
        total_time = sum(
            meta_data[(meta_data["composer"] == composer) & (meta_data["ensemble"] == ensemble)].seconds.tolist())
        print(f"Total time for {composer} with {ensemble} is: {total_time} seconds")

        domain_dir = out_dir / f"{composer}_{ensemble.replace(' ', '_')}"
        if not os.path.exists(domain_dir):
            os.mkdir(domain_dir)

        for fid in fid_list:
            filename = train_dir / f'{fid}.wav'
            if not filename.exists():
                filename = test_dir / f'{fid}.wav'

            copy(str(filename), str(domain_dir))


def main():
    in_dir = Path("../dataset/musicnet")
    out_dir = Path("../dataset/musicnet/parsed")

    out_dir.mkdir(exist_ok=True, parents=True)

    domains = [
        ['Accompanied Violin', 'Beethoven'],
        ['Solo Cello', 'Bach'],
        ['Solo Piano', 'Bach'],
        ['Solo Piano', 'Beethoven'],
        ['String Quartet', 'Beethoven'],
        ['Wind Quintet', 'Cambini'],
    ]

    meta_data = pandas.read_csv(in_dir / 'musicnet_metadata.csv')
    split_domains(domains, in_dir, out_dir, meta_data)


if __name__ == '__main__':
    main()
