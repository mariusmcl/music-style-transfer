import torch.nn as nn


class DomainClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.n_classes = num_classes

        convs = []
        for i in range(config.layers):
            in_channels = config.latent_dim if i == 0 else config.channels
            convs.append(nn.Conv1d(in_channels, config.channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(config.channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=config.dropout_discriminator)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)

        mean = logits.mean(2)
        return mean
