import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.tokenizer.tokenizer import Tokenizer
from src.dataset.dataset import JavaCodeDataset, collate_fn
from src.model.positionalenc import CodeCompletionModel
import pickle
import time

def clean_code(code: str) -> str:
    # Loại bỏ bình luận và khoảng trắng thừa
    lines = [line.strip() for line in code.split("\n") if not line.strip().startswith("//") and line.strip()]
    return "\n".join(lines)

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, patience=10, min_delta=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore_index=0 cho <pad>
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            output = model(input_ids)  # (batch_size, seq_len, vocab_size)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                output = model(input_ids)
                loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break


def generate_code(model, tokenizer, prompt, max_length=50, beam_width=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encodePromt(prompt)], dtype=torch.long).to(device)
    print(f"Input IDs: {input_ids}")

    beams = [(input_ids, 0.0)]  # (sequence, score)

    with torch.no_grad():
        for step in range(max_length):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == tokenizer.token_to_idx['<eos>']:
                    new_beams.append((seq, score))
                    continue

                output = model(seq)
                next_token_logits = output[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                top_probs, top_tokens = torch.topk(probs, beam_width, dim=-1)

                print(f"Step {step + 1}, Top {beam_width} tokens: {[tokenizer.idx_to_token.get(token.item(), '<unk>') for token in top_tokens[0]]}")
                print(f"Probabilities: {top_probs[0].tolist()}")

                for prob, token in zip(top_probs[0], top_tokens[0]):
                    new_seq = torch.cat([seq, token.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            print(f"Step {step + 1} beams: {[(tokenizer.decode(seq[0].tolist()), score) for seq, score in beams]}")

            # Dừng nếu beam có điểm cao nhất kết thúc bằng <eos>
            best_seq, best_score = beams[0]
            if best_seq[0, -1].item() == tokenizer.token_to_idx['<eos>']:
                print(f"Best beam ended with <eos>: {tokenizer.decode(best_seq[0].tolist())}, score: {best_score}")
                break

    best_seq = beams[0][0]
    generated = tokenizer.decode(best_seq[0].tolist())
    print(f"Raw generated IDs: {best_seq[0].tolist()}")
    return generated


def generate_code_faster(model, tokenizer, prompt, max_length=15, device='cuda'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encodePromt(prompt)], dtype=torch.long).to(device)

    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            output = model(generated_ids)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.token_to_idx['<eos>']:
                break

    token_list = generated_ids[0].tolist()
    generated = tokenizer.decode(token_list)
    return generated if generated else "[No valid tokens generated]"

# if __name__ == "__main__":
#     data_dir = r"D:\Desktop\DACN\dacn_1\data\test"
#     for filename in os.listdir(data_dir):
#         with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
#             code = f.read()
#         cleaned_code = clean_code(code)
#         with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as f:
#             f.write(cleaned_code)
#     # Khởi tạo tokenizer
#     tokenizer = Tokenizer()
    
#     dataset = JavaCodeDataset(data_dir, tokenizer, max_length=512)

#     print(f"Total dataset size: {len(dataset)}")
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     print(f"Train size: {train_size}, Validation size: {val_size}")

#     if len(dataset) == 0:
#         raise ValueError("Dataset is empty. Please check the data directory.")

#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

#     vocab_size = len(tokenizer.vocab)
#     model = CodeCompletionModel(
#         vocab_size=vocab_size,
#         d_model=128,  # Giảm từ 256 xuống 128
#         nhead=4,      # Giảm từ 8 xuống 4
#         num_layers=2, # Giảm từ 4 xuống 2
#         dim_feedforward=256,  # Giảm từ 512 xuống 256
#         max_seq_length=512
#     )
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")

#     train_model(model, train_loader, val_loader, num_epochs=50, lr=0.0001, patience=10, min_delta=0.001, device=device)

#     model.load_state_dict(torch.load('best_model.pt'))
#     model.to(device)

#     prompt = "for (int "
#     generated = generate_code(model, tokenizer, prompt, max_length=50, beam_width=5, device=device)
#     print(f"\nPrompt: {prompt}")
#     print(f"Generated: {generated}")

if __name__ == "__main__":
    # Khởi tạo tokenizer và load vocabulary
    data_dir = r"D:\Desktop\DACN\dacn_1\data\test"
    tokenizer = Tokenizer()
    dataset = JavaCodeDataset(data_dir, tokenizer, max_length=128)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Khởi tạo model với cấu hình giống lúc train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CodeCompletionModel(
        vocab_size=vocab_size,
        d_model=128,  # Giảm từ 256 xuống 128
        nhead=4,      # Giảm từ 8 xuống 4
        num_layers=2, # Giảm từ 4 xuống 2
        dim_feedforward=256,  # Giảm từ 512 xuống 256
        max_seq_length=512    # Phải khớp với lúc train
    )
    model.to(device)
    
    # Tải trọng số từ best_model.pt
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    # Vòng lặp nhập prompt và generate code
    print("\nNhập prompt để generate code (nhấn Enter để chạy, nhập 'x' để thoát):")
    while True:
        prompt = input("Prompt: ").strip()
        
        # Nhấn 'x' để thoát
        if prompt.lower() == 'x':
            print("Đã thoát chương trình.")
            break

        # Đo thời gian generate
        start_time = time.time()
        generated = generate_code_faster(model, tokenizer, prompt, max_length=50, device=device)
        end_time = time.time()

        # In kết quả
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
        print(f"Thời gian generate: {end_time - start_time:.4f} giây\n")