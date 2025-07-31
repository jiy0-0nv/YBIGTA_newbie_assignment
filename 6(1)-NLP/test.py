import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.metrics import f1_score

from model import MyGRULanguageModel
from config import *


if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # declare model and load checkpoint
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, vocab_size)
    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint)
    model.to(device)

    # load test dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    test_loader = DataLoader(dataset["test"], batch_size=1)

    # evaluate
    preds = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)
            logits = model(input_ids)

            labels += data["label"].tolist()
            preds += logits.argmax(-1).cpu().tolist()

    # calculate f1 score
    macro = f1_score(labels, preds, average='macro')
    micro = f1_score(labels, preds, average='micro')
    print(f"test result\nmacro: {macro:.6f} | micro: {micro:.6f}")