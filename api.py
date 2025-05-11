from flask import Flask, request, jsonify
import torch
from src.dataset.dataset import JavaCodeDataset
from src.tokenizer.tokenizer import Tokenizer
from src.model.positionalenc import CodeCompletionModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cấu hình thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer và mô hình

data_dir = r"D:\Desktop\DACN\dacn_1\data\test"
tokenizer = Tokenizer()
dataset = JavaCodeDataset(data_dir, tokenizer, max_length=128)
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

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'prompt': prompt, 'generated': ""})

    try:
        # Sinh mã từ prompt
        generated = generate_code_faster(model, tokenizer, prompt, max_length=50, device=device)
        
        # Loại bỏ prompt trong kết quả sinh ra (nếu nó xuất hiện ở đầu)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()  # Cắt phần prompt ra

        return jsonify({'prompt': prompt, 'generated': generated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
