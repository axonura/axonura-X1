# Copyright 2026 First Person
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from torch.utils.data import IterableDataset, DataLoader
from functools import partial
from . import utils


class TextIterDataset(IterableDataset):
    def __init__(self, dataset_shard):
        self.dataset_shard = dataset_shard

    def __iter__(self):
        return utils.text_gen(self.dataset_shard)


class MultimodalIterDataset(IterableDataset):
    def __init__(self, dataset_shard):
        self.dataset_shard = dataset_shard

    def __iter__(self):
        for x in self.dataset_shard:
            yield x


class DSPipeline:
    def __init__(
        self,
        dataset,
        tokenizer,
        max_len=128,
        batch_size=32,
        media_token_ids=(None, None, None),
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.media_token_ids = media_token_ids
        self.multimodal = any(t is not None for t in media_token_ids)
        self.train_data = dataset.get("train", dataset)
        self.test_data = dataset.get("validation", dataset.get("test", None))

    def _make_loader(self, data):
        if self.multimodal:
            ds = MultimodalIterDataset(data)
            collate = partial(
                utils.collate_multimodal,
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                media_token_ids=self.media_token_ids,
            )
        else:
            ds = TextIterDataset(data)
            collate = partial(
                utils.collate_fn, tokenizer=self.tokenizer, max_len=self.max_len
            )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=collate,
            drop_last=True,
        )

    def call(self):
        train_loader = self._make_loader(self.train_data)

        val_loader = None
        if self.test_data:
            val_loader = self._make_loader(self.test_data)

        return train_loader, val_loader
