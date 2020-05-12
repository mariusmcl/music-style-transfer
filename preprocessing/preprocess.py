from pathlib import Path
from dataset_factory import EncodedFilesDataset


def main():
    in_dir = Path("../dataset/musicnet/split")
    out_dir = Path("../dataset/musicnet/preprocessed")
    should_normalize = False

    dataset = EncodedFilesDataset(in_dir)
    dataset.dump_to_folder(out_dir, norm_db=should_normalize)


if __name__ == '__main__':
    main()
