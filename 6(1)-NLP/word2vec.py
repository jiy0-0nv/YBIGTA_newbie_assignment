import time
import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!
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
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        self.num_negatives = num_negatives
        if method not in ("cbow", "skipgram"):
            raise ValueError("method must be 'cbow' or 'skipgram'")

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!

        print("▶ fit 시작")
        pad_id = tokenizer.pad_token_id
        corpus_ids = [
            tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )
            for text in corpus
        ]

        # negative sampling 분포 준비
        all_tokens = [t for sent in corpus_ids for t in sent if t != pad_id]
        freqs = Counter(all_tokens)
        dist = torch.zeros(self.embeddings.num_embeddings)
        for t,c in freqs.items():
            dist[t] = c**0.75
        dist /= dist.sum()
        self.register_buffer("unigram_dist", dist)

        total_sentences = len(corpus_ids)
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0

            for idx, sentence in enumerate(corpus_ids, start=1):
                if len(sentence) < 2:
                    continue

                if self.method == "cbow":
                    loss = self._train_cbow(sentence, pad_id, criterion, optimizer)
                else:
                    loss = self._train_skipgram(sentence, pad_id, criterion, optimizer)
                epoch_loss += loss

                # 10문장마다 진행 상황 출력
                if idx % 10 == 0:
                    elapsed = time.time() - epoch_start
                    print(
                        f"[Epoch {epoch}] {idx}/{total_sentences} sentences, "
                        f"elapsed {elapsed:.1f}s",
                        flush=True
                    )

            avg_loss = epoch_loss / total_sentences
            epoch_time = time.time() - epoch_start
            print(
                f"✔ Epoch {epoch}/{num_epochs} 완료 "
                f"(time: {epoch_time:.1f}s, avg loss: {avg_loss:.4f})"
            )

    def _train_cbow(
        self,
        # 구현하세요!
        sentence: list[int],
        pad_id: int | None,
        criterion,
        optimizer
    ) -> float:
        # 구현하세요!
        losses: list[float] = []
        for idx, target in enumerate(sentence):
            if pad_id is not None and target == pad_id:
                continue
            start = max(0, idx - self.window_size)
            end = min(len(sentence), idx + self.window_size + 1)
            contexts = [
                sentence[j]
                for j in range(start, end)
                if j != idx and (pad_id is None or sentence[j] != pad_id)
            ]
            if not contexts:
                continue

            ctx_tensor = torch.LongTensor(contexts)
            ctx_emb = self.embeddings(ctx_tensor)
            ctx_mean = ctx_emb.mean(dim=0, keepdim=True)
            logits = self.weight(ctx_mean)
            tgt_tensor = torch.LongTensor([target])

            optimizer.zero_grad()
            loss = criterion(logits, tgt_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses) if losses else 0.0

    def _train_skipgram(
        self,
        # 구현하세요!
        sentence: list[int],
        pad_id: int | None,
        criterion,
        optimizer
    ) -> float:
        # 구현하세요!
        device = next(self.parameters()).device
        losses = []
        for idx, target in enumerate(sentence):
            if pad_id is not None and target == pad_id: continue
            start = max(0, idx - self.window_size)
            end   = min(len(sentence), idx + self.window_size + 1)
            for j in range(start, end):
                if j == idx: continue
                context = sentence[j]
                if pad_id is not None and context == pad_id: continue

                # positive pair
                tgt = torch.tensor([target], device=device)
                ctx = torch.tensor([context], device=device)
                input_e = self.embeddings(tgt).squeeze(0)
                pos_e   = self.weight.weight[ctx].squeeze(0)

                # negative samples
                neg_ids = torch.multinomial(
                    self.unigram_dist, 
                    self.num_negatives, 
                    replacement=True
                ).to(device)
                neg_e = self.weight.weight[neg_ids]

                # score 계산
                pos_score = torch.dot(input_e, pos_e)
                neg_score = neg_e @ input_e

                # negative sampling loss
                loss = - (
                    F.logsigmoid(pos_score)
                    + torch.sum(F.logsigmoid(-neg_score))
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return sum(losses) / len(losses) if losses else 0.0