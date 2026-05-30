import torch


def text_gen(dataset_shard):
    for x in dataset_shard:
        text = x.get("text", "")
        if text and text.strip():
            yield text


def encode_batch(texts, tokenizer, max_len):
    texts = [t if t.strip() else " " for t in texts]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"]


def collate_fn(batch, tokenizer, max_len):
    input_ids = encode_batch(batch, tokenizer, max_len)
    return input_ids[:, :-1], input_ids[:, 1:]


def batch_iterator(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


def top_k_sampling(logits, k=64):
    top_k_vals, top_k_indices = torch.topk(logits, k=k, dim=-1)
    top_k_logits = torch.full_like(logits, -1e9)
    top_k_logits.scatter_(-1, top_k_indices, top_k_vals)
    probs = torch.softmax(top_k_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
