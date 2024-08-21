# %%
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Literal, Optional

import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from utils.clogging import getColoredLogger
from utils.initialize import initializer

import wandb

logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")
project_root_dir = initializer(globals(), logger=logger)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wandb.init(project="transformer-tutorial", name="de-en-transformer")


def assert_size_of_tensor(name: str, tensor: Optional[torch.Tensor], *, expected_size: tuple[int, ...], optional=True) -> None:
    if optional and tensor is None:
        return
    else:
        assert tensor.size() == torch.Size(expected_size), f"tensor {name} size is {tensor.size()}, expected {expected_size}"


class PositionalEncoding(nn.Module):
    """Positional encoding

    .forward()
    Args:
        token_embeddings (torch.Tensor): Token embeddings

    Returns:
        torch.Tensor: Positional encoded token embeddings
    """

    def __init__(self, embed_dim: int = 512, dropout_p: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.max_len = max_len
        p = torch.arange(0, max_len).reshape(max_len, 1)
        d = torch.exp(-torch.arange(0, embed_dim, 2) * (torch.log(torch.tensor(10000.0)) / embed_dim))

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(p * d)
        pe[:, 1::2] = torch.cos(p * d)
        pe = pe.unsqueeze(-2)
        assert_size_of_tensor("pe", pe, expected_size=(max_len, 1, embed_dim))
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.dropout(token_embeddings + self.pe[: token_embeddings.size(0)])


class TokenEmbedding(nn.Module):
    """Token embedding

    .forward()
    Args:
        x (torch.Tensor[torch.long]): Token indices

    Returns:
        torch.Tensor: Token embeddings
    """

    def __init__(self, vocab_size: int, embed_dim: int = 512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * (self.embed_dim**0.5)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 20000,
        tgt_vocab_size: int = 20000,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        emb_size: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            batch_first=True,
        )
        self.head = nn.Linear(emb_size, tgt_vocab_size)
        self.source_token_embedding = TokenEmbedding(src_vocab_size, emb_size)
        self.target_token_embedding = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout_p=dropout_p)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        source_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        source_embeddings = self.positional_encoding(token_embeddings=self.source_token_embedding(tokens=source_tokens))
        target_embeddings = self.positional_encoding(token_embeddings=self.target_token_embedding(tokens=target_tokens))

        outs = self.transformer(
            src=source_embeddings,
            tgt=target_embeddings,
            src_mask=source_mask,
            tgt_mask=target_mask,
            memory_mask=None,
            src_key_padding_mask=source_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            src_is_causal=None,
            tgt_is_causal=None,
            memory_is_causal=None,
        )
        outs = outs.contiguous()
        return self.head(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.source_token_embedding(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.target_token_embedding(tgt)), memory, tgt_mask)

    def predict(self, source: torch.Tensor, source_mask: torch.Tensor, max_len: int = 100, bos_idx: int = 2, eos_idx: int = 3):
        self.eval()
        with torch.inference_mode():
            memory = self.encode(source, source_mask)
            target = torch.tensor([bos_idx]).type_as(source).to(source.device)
            for i in range(max_len - 1):
                target_mask = (generate_square_subsequent_mask(len(target)).type(torch.bool)).to(target.device)
                output = self.decode(target.unsqueeze(0), memory, target_mask)
                prob = self.head(output[:, -1])
                next_word_id = prob.argmax(dim=1)
                target = torch.cat((target, next_word_id), dim=0)
                if next_word_id == eos_idx:
                    break
        if target[-1] != eos_idx:
            target = torch.cat((target, torch.tensor([eos_idx]).type_as(source).to(source.device)), dim=0)
        return target


def generate_square_subsequent_mask(sequence_length: int):
    mask = (torch.triu(torch.ones((sequence_length, sequence_length))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 1):
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(src.device)

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class SpaCyTokenizer:
    def __init__(self, language: str, special_tokens=("<unk>", "<pad>", "<bos>", "<eos>")):
        self.nlp = spacy.load(language)
        self.special_tokens = special_tokens
        self.unk_id = 0
        self.pad_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.token_to_id = {k: i for i, k in enumerate(self.special_tokens)}
        self.id_to_token = {i: k for i, k in enumerate(self.special_tokens)}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def build(self, train_sentences: list[str]):
        counter = Counter()
        for sentence in train_sentences:
            counter.update(self(sentence.strip(), add_bos_eos_token=False))

        self.token_to_id.update(
            {
                k: i
                for i, (k, v) in enumerate(
                    list(sorted(OrderedDict(counter).items(), key=lambda x: x[1], reverse=True)),
                    start=len(self.token_to_id) + 1,
                )
            }
        )
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __call__(self, text: str, add_bos_eos_token: bool = True) -> list[str]:
        if add_bos_eos_token:
            return ["<bos>"] + [token.text for token in self.nlp.tokenizer(text)] + ["<eos>"]
        return [token.text for token in self.nlp.tokenizer(text)]

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in self(text)], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        return [
            self.id_to_token[token_id]
            for token_id in token_ids.tolist()
            if not skip_special_tokens or token_id not in self.special_tokens
        ]

    def batch_encode(self, sentences: list[str]) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            [self.encode(sentence) for sentence in sentences],
            batch_first=True,
            padding_value=self.pad_id,
        )

    def batch_decode(
        self, encoded_sentences: torch.Tensor | list[torch.Tensor], skip_special_tokens: bool = True
    ) -> list[list[str]]:
        if isinstance(encoded_sentences, torch.Tensor):
            encoded_sentences = encoded_sentences.tolist()
        return [
            [
                self.id_to_token[token_id]
                for token_id in encoded_sentence
                if not skip_special_tokens or token_id not in self.special_tokens
            ]
            for encoded_sentence in encoded_sentences
        ]


class TextPairDataset(Dataset):
    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        source_tokenizer: Callable[[str], list[str]],
        target_tokenizer: Callable[[str], list[str]],
    ):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        assert len(source_sentences) == len(target_sentences), "source and target sentences must have the same length"
        self.length = len(source_sentences)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.source_sentences[index], self.target_sentences[index]

    def default_tokenize_collate_fn(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        source_sentences, target_sentences = zip(*batch)
        return self.source_tokenizer.batch_encode(source_sentences), self.target_tokenizer.batch_encode(target_sentences)


def train(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    pad_idx: int,
    device_type: Literal["cpu", "cuda"] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler()
    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)
        # source: (batch_size, source_seq_len), target: (batch_size, target_seq_len)

        target_input = target[:, :-1]
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_input, pad_idx)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            logits = model(
                source_tokens=source,
                target_tokens=target_input,
                source_mask=source_mask,
                target_mask=target_mask,
                source_key_padding_mask=source_padding_mask,
                target_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=source_padding_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if i % 100 == 0:
            logger.info(f"Batch {i}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    pad_idx: int,
    device_type: Literal["cpu", "cuda"] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    model.eval()
    total_loss = 0
    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)
        target_input = target[:, :-1]
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_input, pad_idx)

        with torch.inference_mode():
            logits = model(
                source_tokens=source,
                target_tokens=target_input,
                source_mask=source_mask,
                target_mask=target_mask,
                source_key_padding_mask=source_padding_mask,
                target_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=source_padding_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
        total_loss += loss.item()

    return total_loss / len(dataloader)


def translate(
    model: Seq2SeqTransformer,
    source_sentence: str,
    source_tokenizer: SpaCyTokenizer,
    target_tokenizer: SpaCyTokenizer,
    max_length: int = 100,
    device: torch.device = torch.device("cpu"),
):
    model.eval()
    source_token = source_tokenizer.encode(source_sentence).unsqueeze(0).to(device)
    source_mask = (torch.zeros(source_token.size(1), source_token.size(1))).type(torch.bool).to(device)
    target_token = model.predict(
        source=source_token,
        source_mask=source_mask,
        max_len=max_length,
        bos_idx=source_tokenizer.bos_id,
        eos_idx=source_tokenizer.eos_id,
    )
    target_sentence = target_tokenizer.decode(target_token, skip_special_tokens=True)
    return target_sentence


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 64
    source_language_name = "de"
    target_language_name = "en"
    wandb.config.update(
        {"source_language": source_language_name, "target_language": target_language_name, "batch_size": batch_size}
    )

    dataset_dir = Path.home() / "datasets" / "Multi30k"
    model_dir = project_root_dir / "models" / "seq2seq_tutorial_deen"
    model_dir.mkdir(parents=True, exist_ok=True)
    # must be downloaded
    train_source = dataset_dir / f"train.{source_language_name}"
    train_target = dataset_dir / f"train.{target_language_name}"
    val_source = dataset_dir / f"val.{source_language_name}"
    val_target = dataset_dir / f"val.{target_language_name}"

    with open(train_source) as f, open(train_target) as g:
        train_source_sentences = tuple(map(lambda x: x.strip(), f.readlines()))
        train_target_sentences = tuple(map(lambda x: x.strip(), g.readlines()))

    with open(val_source) as f, open(val_target) as g:
        val_source_sentences = tuple(map(lambda x: x.strip(), f.readlines()))
        val_target_sentences = tuple(map(lambda x: x.strip(), g.readlines()))

    source_tokenizer = SpaCyTokenizer("de_core_news_sm")
    target_tokenizer = SpaCyTokenizer("en_core_web_sm")
    source_tokenizer.build(train_source_sentences)
    target_tokenizer.build(train_target_sentences)

    train_dataset = TextPairDataset(
        source_sentences=train_source_sentences,
        target_sentences=train_target_sentences,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=train_dataset.default_tokenize_collate_fn, shuffle=True
    )

    val_dataset = TextPairDataset(
        source_sentences=val_source_sentences,
        target_sentences=val_target_sentences,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=val_dataset.default_tokenize_collate_fn, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Training
    model_kwargs = dict(
        src_vocab_size=source_tokenizer.vocab_size + 1,
        tgt_vocab_size=target_tokenizer.vocab_size + 1,
        num_encoder_layers=4,
        num_decoder_layers=4,
        emb_size=256,
        nhead=8,
        dim_feedforward=256,
        dropout_p=0.25,
    )
    wandb.config.update(model_kwargs)

    model = Seq2SeqTransformer(**model_kwargs)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=source_tokenizer.pad_id)
    epochs = 100

    best_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        sample_table = wandb.Table(columns=["source", "target", "prediction"])
        for sample_idx in range(8):
            sample_source, sample_target = val_dataset[sample_idx]
            sample_pred = translate(
                model=model,
                source_sentence=sample_source,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                max_length=128,
                device=device,
            )
            sample_table.add_data(sample_source, sample_target, " ".join(sample_pred))

        wandb.log({f"sample_translation_{epoch}": sample_table})

        train_loss = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_idx=source_tokenizer.pad_id,
        )
        valid_loss = validate(
            model=model,
            dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_idx=source_tokenizer.pad_id,
        )
        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "lr": optimizer.param_groups[0]["lr"]})

        if train_loss < best_loss:
            best_loss = train_loss
            best_model = model.state_dict()
            torch.save(
                {
                    "model_state_dict": best_model,
                    "source_tokenizer": source_tokenizer,
                    "target_tokenizer": target_tokenizer,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                model_dir / "best_model.pth",
            )

        scheduler.step()

wandb.finish()

# %%
