import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

from pathlib import Path
import torch
import matplotlib
import h5py
import tqdm

from config import Map
from utils.helper_functions import save_audio
from models.decoder import Decoder
from models.encoder import Encoder
from models.sample_generator import SampleGenerator
import utils.helper_functions as helper_functions
import librosa

matplotlib.use('agg')


def extract_id(path):
    decoder_id = str(path)[:-4].split('_')[-1]
    return int(decoder_id)


def main(config):
    print('Starting')

    checkpoints = config.checkpoint.parent.glob(config.checkpoint.name + '_*.pth')
    checkpoints = [c for c in checkpoints if extract_id(c) in config.decoders]
    assert len(checkpoints) >= 1, "No checkpoints found."

    model_config = torch.load(config.checkpoint.parent / 'args.pth')[0]
    encoder = Encoder(model_config.encoder)
    encoder.load_state_dict(torch.load(checkpoints[0])['encoder_state'])
    encoder.eval()
    encoder = encoder.cuda()

    generators = []
    generator_ids = []
    for checkpoint in checkpoints:
        decoder = Decoder(model_config.decoder)
        decoder.load_state_dict(torch.load(checkpoint)['decoder_state'])
        decoder.eval()
        decoder = decoder.cuda()

        generator = SampleGenerator(decoder, config.batch_size, wav_freq=config.rate)

        generators.append(generator)
        generator_ids.append(extract_id(checkpoint))

    xs = []
    assert config.out_dir is not None

    if len(config.sample_dir) == 1 and config.sample_dir[0].is_dir():
        top = config.sample_dir[0]
        file_paths = list(top.glob('**/*.wav')) + list(top.glob('**/*.h5'))
    else:
        file_paths = config.sample_dir

    print("File paths to be used:", file_paths)
    for file_path in file_paths:
        if file_path.suffix == '.wav':
            data, rate = librosa.load(file_path, sr=config.rate)
            data = helper_functions.mu_law(data)
        elif file_path.suffix == '.h5':
            data = helper_functions.mu_law(h5py.File(file_path, 'r')['wav'][:] / (2 ** 15))
            if data.shape[-1] % config.rate != 0:
                data = data[:-(data.shape[-1] % config.rate)]
            assert data.shape[-1] % config.rate == 0
            print(data.shape)
        else:
            raise Exception(f'Unsupported filetype {file_path}')

        if config.sample_len:
            data = data[:config.sample_len]
        else:
            config.sample_len = len(data)
        xs.append(torch.tensor(data).unsqueeze(0).float().cuda())

    xs = torch.stack(xs).contiguous()
    print(f'xs size: {xs.size()}')

    def save(x, decoder_idx, filepath):
        wav = helper_functions.inv_mu_law(x.cpu().numpy())
        print(f'X size: {x.shape}')
        print(f'X min: {x.min()}, max: {x.max()}')

        save_audio(wav.squeeze(), config.out_dir / str(decoder_idx) / filepath.with_suffix('.wav').name,
                   rate=config.rate)

    yy = {}
    with torch.no_grad():
        zz = []
        for xs_batch in torch.split(xs, config.batch_size):
            zz += [encoder(xs_batch)]
        zz = torch.cat(zz, dim=0)

        for i, generator_id in enumerate(generator_ids):
            yy[generator_id] = []
            generator = generators[i]
            for zz_batch in torch.split(zz, config.batch_size):
                print("Batch shape:", zz_batch.shape)
                splits = torch.split(zz_batch, config.split_size, -1)
                audio_data = []
                generator.reset()
                for cond in tqdm.tqdm(splits):
                    audio_data += [generator.generate(cond).cpu()]
                audio_data = torch.cat(audio_data, -1)
                yy[generator_id] += [audio_data]
            yy[generator_id] = torch.cat(yy[generator_id], dim=0)

            for sample_result, filepath in zip(yy[generator_id], file_paths):
                save(sample_result, generator_id, filepath)

            del generator


if __name__ == '__main__':
    config = Map({
        "sample_dir": [Path("dataset/samples/input")],
        "out_dir": Path("dataset/samples/output"),
        "checkpoint": Path("checkpoints/trained_models/lastmodel"),
        "decoders": [0, 1],
        "rate": 16_000,
        "batch_size": 5,
        "sample_len": 16_000*10,
        "split_size": 20,
    })
    main(config)
