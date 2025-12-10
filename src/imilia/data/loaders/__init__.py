"""Data loaders package for IBDColEPI dataset."""

from imilia.data.loaders.ibdcolepi_dataloader import IBDColEPIDataLoader
from imilia.data.loaders.loader_histo import load_data, IBDColEpiHistoLoader

__all__ = [
    "IBDColEPIDataLoader",
    "IBDColEpiHistoLoader",
    "load_data",
]
