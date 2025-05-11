import re
from collections import Counter
from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class Tokenizer:
    def __init__(self):
        self.tokenizer = HuggingFaceTokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", " "]
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.tokenizer.post_processor = TemplateProcessing(
            single="<sos> $A <eos>",
            special_tokens=[("<sos>", self.special_tokens.index("<sos>")), 
                           ("<eos>", self.special_tokens.index("<eos>"))]
        )

    def build_vocab(self, train_data):
        cleaned_data = [self.remove_comments(code) for code in train_data if code.strip()]
        trainer = BpeTrainer(
            vocab_size=15000,
            special_tokens=self.special_tokens,
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(cleaned_data, trainer)
        self.vocab = self.tokenizer.get_vocab()
        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        if len(self.vocab) <= len(self.special_tokens):
            raise ValueError("Vocabulary is empty. No valid tokens found in the data.")

    def remove_comments(self, code: str) -> str:
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*?\n', '\n', code)
        return code

    def tokenize(self, code: str) -> list[str]:
        cleaned_code = self.remove_comments(code)
        if not cleaned_code.strip():
            return []
        encoded = self.tokenizer.encode(cleaned_code)
        tokens = encoded.tokens
        if tokens and tokens[0] == "<sos>":
            tokens = tokens[1:]
        if tokens and tokens[-1] == "<eos>":
            tokens = tokens[:-1]
        return tokens

    def encode(self, code: str) -> list[int]:
        cleaned_code = self.remove_comments(code)
        if not cleaned_code.strip():
            return [self.token_to_idx["<sos>"], self.token_to_idx["<eos>"]]
        encoded = self.tokenizer.encode(cleaned_code)
        return encoded.ids

    def encodePromt(self, code: str) -> list[int]:
        cleaned_code = self.remove_comments(code)
        if not cleaned_code.strip():
            return [self.token_to_idx["<sos>"]]
        tokens = ["<sos>"] + self.tokenize(cleaned_code)
        unk_id = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(token, unk_id) for token in tokens]

    def decode(self, token_idx: list[int]) -> str:
        tokens = [self.idx_to_token.get(idx, "<unk>") for idx in token_idx]
        if "<eos>" in tokens:
            tokens = tokens[:tokens.index("<eos>")]
        if "<sos>" in tokens:
            tokens = tokens[tokens.index("<sos>") + 1:]
        result = ""
        for token in tokens:
            if token in ["<pad>", "<unk>", "<sos>", "<eos>"]:
                continue
            result += token
        return result if result else "[No valid tokens generated]"

# Đoạn mã thử nghiệm
if __name__ == "__main__":
    # Tạo tokenizer
    tokenizer = Tokenizer()
    
    # Dữ liệu giả lập để huấn luyện BPE
    train_data = [
        "Scanner sc = new Scanner(System.in);",
        "for (int i = 0; i < n; i++) {}",
        "int n = 5;",
        "for (int j = 0; j < n; j++) {}",
        "Scanner scanner = new Scanner(System.in);",
        "Scanner scanner = new Scanner(System.in);",
        "Scanner scanner = new Scanner(System.in);"
    ]
    tokenizer.build_vocab(train_data)
    
    # Đoạn mã cần kiểm tra
    test_code = "Scanner sc = new Scanner(System.in); for (int i = 0; i < n; i++) {}"
    
    # Tokenize và in kết quả
    tokens = tokenizer.tokenize(test_code)
    print("Tokens:", tokens)
    
    # Encode và in kết quả
    encoded = tokenizer.encode(test_code)
    print("Encoded:", encoded)
    
    # Decode để kiểm tra lại
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)