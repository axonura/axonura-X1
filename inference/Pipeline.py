from torch.utils.data import IterableDataset, DataLoader
from functools import partial
from . import utils


class TextIterDataset(IterableDataset):
    def __init__(self, dataset_shard):
        self.dataset_shard = dataset_shard

    def __iter__(self):
        return utils.text_gen(self.dataset_shard)


class DSPipeline:
    def __init__(self, dataset, tokenizer, max_len=128, batch_size=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_data = dataset.get("train", dataset)
        self.test_data = dataset.get("validation", dataset.get("test", None))

    def call(self):
        train_ds = TextIterDataset(self.train_data)
        collate = partial(utils.collate_fn, tokenizer=self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            collate_fn=collate,
            drop_last=True,
        )

        val_loader = None
        if self.test_data:
            val_ds = TextIterDataset(self.test_data)
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                collate_fn=collate,
                drop_last=True,
            )

        return train_loader, val_loader
