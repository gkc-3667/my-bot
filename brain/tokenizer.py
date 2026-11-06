"""A tiny reversible byte tokenizer; it uses no pretrained vocabulary.

Each UTF-8 byte is one token (0..255). Special tokens are:
BOS=256, EOS=257, PAD=258. This is intentionally simple for the
first training pipeline; a learned BPE tokenizer can replace it later.
"""

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259


class ByteTokenizer:
    def encode(self, text, add_special_tokens=True):
        token_ids = list(text.encode("utf-8"))
        if add_special_tokens:
            return [BOS_ID, *token_ids, EOS_ID]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        data = bytearray()
        for token_id in token_ids:
            token_id = int(token_id)
            if 0 <= token_id <= 255:
                data.append(token_id)
            elif not skip_special_tokens:
                marker = {
                    BOS_ID: "<BOS>",
                    EOS_ID: "<EOS>",
                    PAD_ID: "<PAD>",
                }.get(token_id, "<UNK>")
                data.extend(marker.encode("utf-8"))
        return data.decode("utf-8", errors="replace")
