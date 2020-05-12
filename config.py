from pathlib import Path

data_path = Path("dataset/musicnet/preprocessed")


class Map(dict):
    """
    Example usage:
    c = Map({'first_name': 'Henrik'})
    """

    def __init__(self, dictionary: dict):
        super(Map, self).__init__()
        if isinstance(dictionary, dict):
            for k, v in dictionary.items():
                if isinstance(v, dict):
                    self[k] = Map(v)
                else:
                    self[k] = v
        else:
            raise NotImplementedError

    def __getattr__(self, attr):
        val = self.get(attr)
        if val is None:
            raise Exception("Key not present in map")

        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)



config = Map({
    "env": {
        "rank": 0,
        "epochs": 500,
        "seed": 1,
        "save_per_epoch": False,
        "data_path": data_path,
        "checkpoint": False,  # Path("checkpoints/pretrained_models/lastmodel"),
        "expPath": Path("checkpoints/trained_models_l_dim_small"),
        "load_optimizer": False,
    },
    "data": {
        "datasets": [
            data_path / "Bach_Solo_Cello",
            data_path / "Beethoven_Solo_Piano",
        ],
        "seq_len": 12_000,
        "epoch_len": 500,  # originally 10_000, but we use 2 datasets instead of 6
        "batch_size": 12,
        "num_workers": 8,
        "data_aug": True,
        "magnitude": 0.5,
        "lr": 1e-3,
        "lr_decay": 0.995,
        "short": False,
        "h5_dataset_name": "wav",
    },
    "encoder": {
        "hidden_layers": 6,
        "channels": 128,
        "blocks": 3,
        "pool": 800,
        "kernel_size": 1,
        "layers": 10,
        "func": "relu",
        "latent_dim": 16,
    },
    "decoder": {
        "blocks": 4,
        "layers": 14,
        "kernel_size": 2,
        "residual_channels": 128,
        "skip_channels": 128,
        "latent_dim": 16,

    },
    "domain_classifier": {
        "layers": 3,
        "channels": 100,
        "condition_dim": 1024,
        "d_lambda": 1e-2,
        "dropout_discriminator": 0.0,
        "grad_clip": 1,
        "latent_dim": 16,
    }
})
