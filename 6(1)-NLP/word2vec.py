import time
import torch
from torch import nn, Tensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal, Optional
from collections import Counter
import torch.nn.functional as F

class Word2Vec(nn.Module):
    unigram_dist: Tensor

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"],
        num_negatives: int = 5
    ) -> None:
        super().__init__()
        if method not in ('cbow', 'skipgram'):
            raise ValueError("method must be 'cbow' or 'skipgram'")
        
        self.embeddings    = nn.Embedding(vocab_size, d_model)
        self.weight        = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size   = window_size
        self.method        = method
        self.num_negatives = num_negatives

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int,
        max_sentences: Optional[int] = None
    ) -> None:

        pad_id = tokenizer.pad_token_id
        corpus = corpus[:max_sentences] if max_sentences else corpus
        corpus_ids = [
            tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
            for text in corpus
        ]
        
        tokens = [t for sent in corpus_ids for t in sent if t != pad_id]
        freqs = Counter(tokens)
        dist = torch.zeros(self.embeddings.num_embeddings)
        for t, c in freqs.items():
            dist[t] = c ** 0.75
        dist /= dist.sum()
        self.register_buffer("unigram_dist", dist)

        optimizer = Adam(self.parameters(), lr=lr)

        total_sents = len(corpus_ids)
        print(f"Training start: {total_sents} sentences, {num_epochs} epochs")
        for epoch in range(1, num_epochs+1):
            start_time = time.time()
            epoch_loss = 0.0
            for idx, sent in enumerate(corpus_ids, 1):
                if len(sent) < 2:
                    continue
                if self.method == 'cbow':
                    loss = self._train_cbow(sent, pad_id, optimizer)
                else:
                    loss = self._train_skipgram(sent, pad_id, optimizer)
                epoch_loss += loss
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f" [Epoch {epoch}] {idx}/{total_sents} sents, {elapsed:.1f}s", flush=True)
            avg_loss = epoch_loss / total_sents
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} done (time {elapsed:.1f}s, avg_loss {avg_loss:.4f})")

    def _train_cbow(
        self,
        sentence: list[int],
        pad_id: Optional[int],
        optimizer
    ) -> float:
        device = next(self.parameters()).device
        total_loss = torch.zeros((), device=device)
        n = 0
        for i, target in enumerate(sentence):
            if pad_id is not None and target == pad_id:
                continue
            start = max(0, i - self.window_size)
            end   = min(len(sentence), i + self.window_size + 1)
            contexts = [sentence[j] for j in range(start, end)
                        if j != i and (pad_id is None or sentence[j] != pad_id)]
            if not contexts:
                continue
            ctx_ids = torch.tensor(contexts, device=device)
            ctx_emb = self.embeddings(ctx_ids)
            ctx_mean = ctx_emb.mean(dim=0, keepdim=True)
            target_id = torch.tensor([target], device=device)
            logits = self.weight(ctx_mean)
            loss = F.cross_entropy(logits, target_id)
            total_loss += loss
            n += 1
        if n == 0:
            return 0.0
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return (total_loss.item() / n)

    def _train_skipgram(
        self,
        sentence: list[int],
        pad_id: Optional[int],
        optimizer
    ) -> float:
        device = next(self.parameters()).device
        total_loss = torch.zeros((), device=device)
        n = 0
        for i, tgt in enumerate(sentence):
            if pad_id is not None and tgt == pad_id:
                continue
            start = max(0, i - self.window_size)
            end   = min(len(sentence), i + self.window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                ctx = sentence[j]
                if pad_id is not None and ctx == pad_id:
                    continue
                # embeddings
                tgt_emb = self.embeddings(torch.tensor(tgt, device=device))  # (d,)
                pos_emb = self.weight.weight[ctx]                            # (d,)
                # negative sampling ids
                neg_ids = torch.multinomial(self.unigram_dist, self.num_negatives, replacement=True).to(device)
                neg_emb = self.weight.weight[neg_ids]                        # (K,d)
                # scores
                pos_score = torch.dot(tgt_emb, pos_emb)
                neg_score = neg_emb @ tgt_emb
                loss = - (F.logsigmoid(pos_score) + torch.sum(F.logsigmoid(-neg_score)))
                total_loss += loss
                n += 1
        if n == 0:
            return 0.0
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return (total_loss.item() / n)
