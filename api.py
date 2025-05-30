from flask import Flask, request, jsonify
import torch
from src.dataset.dataset import JavaCodeDataset
from src.tokenizer.tokenizer import Tokenizer
from src.model.positionalenc import CodeCompletionModel
from flask_cors import CORS
import re
import difflib

app = Flask(__name__)
CORS(app)

# Cấu hình thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer và mô hình

data_dir = r"D:\Desktop\DACN\dacn_1\data\p02621"
tokenizer = Tokenizer()
dataset = JavaCodeDataset(data_dir, tokenizer, max_length=512)
vocab_size = len(tokenizer.vocab)

model = CodeCompletionModel(
    vocab_size=vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    max_seq_length=512
)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

def generate_code_faster(model, tokenizer, prompt, max_length=50, device='cuda'):
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
    return tokenizer.decode(generated_ids[0].tolist())

def generate_code_top1(model, tokenizer, prompt, max_length=50, top_k=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encodePromt(prompt)], dtype=torch.long).to(device)

    beams = [(input_ids, 0.0)]  # (sequence, score)

    with torch.no_grad():
        for step in range(max_length):
            new_beams = []
            step_tokens_log = []
            step_probs_log = []

            for seq, score in beams:
                if seq[0, -1].item() == tokenizer.token_to_idx['<eos>']:
                    new_beams.append((seq, score))
                    continue

                output = model(seq)
                next_token_logits = output[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                top_probs, top_tokens = torch.topk(probs, k=5, dim=-1)  # Lấy top 5 để log thôi, không ảnh hưởng beam

                tokens_str = [tokenizer.idx_to_token.get(token.item(), '<unk>') for token in top_tokens[0]]
                step_tokens_log.append(tokens_str)
                step_probs_log.append(top_probs[0].tolist())

                for prob, token in zip(top_probs[0], top_tokens[0]):
                    new_seq = torch.cat([seq, token.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_seq, new_score))

            # Giữ duy nhất 1 beam có score cao nhất
            best_beam = max(new_beams, key=lambda x: x[1])
            beams = [best_beam]

    # Trả về top_k kết quả từ beam cuối cùng (ở đây thực tế chỉ có 1 beam duy nhất)
    results = []
    final_beam = beams[0]
    decoded = tokenizer.decode(final_beam[0][0].tolist())
    results.append((decoded, final_beam[1]))

    return decoded

def generate_code(model, tokenizer, prompt, max_length=100, beam_width=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encodePromt(prompt)], dtype=torch.long).to(device)

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

                for prob, token in zip(top_probs[0], top_tokens[0]):
                    new_seq = torch.cat([seq, token.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Dừng nếu beam có điểm cao nhất kết thúc bằng <eos>
            best_seq, best_score = beams[0]
            if best_seq[0, -1].item() == tokenizer.token_to_idx['<eos>']:
                break

    best_seq = beams[0][0]
    generated = tokenizer.decode(best_seq[0].tolist())
    return generated


def is_java_function_declaration(line: str) -> bool:
    """
    Kiểm tra một dòng code có phải là khai báo hàm trong Java không.
    Hỗ trợ khai báo với phạm vi truy cập (public, private, protected) và kiểu trả về.

    Ví dụ:
        public void sayHello() {
        private int add(int a, int b) {
        static String getName() {
    """
    line = line.strip()

    # Regex kiểm tra khai báo hàm Java
    pattern = r'^(public|private|protected)?\s*(static\s+)?[\w<>]+\s+\w+\s*\([^)]*\)\s*\{?$'
    
    return bool(re.match(pattern, line))

def remove_prompt_from_generated(prompt, generated):
    matcher = difflib.SequenceMatcher(None, prompt, generated)
    match = matcher.find_longest_match(0, len(prompt), 0, len(generated))

    # Nếu phần khớp nằm ở đầu của cả 2 chuỗi
    if match.a == 0 and match.b == 0:
        result = generated[match.size:].lstrip()
    else:
        result = generated

    # Loại bỏ dấu '}' nếu nó nằm ở cuối
    if result.endswith('}'):
        result = result[:-1].rstrip()

    return result

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    current_line = data.get('currentLine', '').strip()
    pre_line = data.get('prevLine', '').strip()
    print("Current:", current_line)
    print("Pre:", pre_line)

    if not current_line and not pre_line:
        print("da gui")
        return jsonify({'prompt': "", 'generated': ""})

    try:
        if is_java_function_declaration(pre_line):
            prompt = "<block> " + pre_line + " " + current_line
            cutPrompt = pre_line + " " + current_line
        else:
            prompt = "<line> " + current_line
            cutPrompt = current_line
        # Sinh mã từ prompt
        print(prompt)
        generated = generate_code(model, tokenizer, prompt, max_length=100, device=device)
        print(generated)
        
        cutPrompt = cutPrompt.strip()
        generated_clean = generated.strip()
        
        # Loại bỏ prompt trong kết quả sinh ra (nếu nó xuất hiện ở đầu)
        gen = remove_prompt_from_generated(cutPrompt, generated_clean)

        return jsonify({'prompt': cutPrompt, 'generated': gen})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
