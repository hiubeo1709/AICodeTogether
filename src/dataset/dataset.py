import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.tokenizer.tokenizer import Tokenizer
import re

class JavaCodeDataset(Dataset):
    def __init__(self, directory, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load dữ liệu từ thư mụcD
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                            nor_code_format = normalize_code_format(code)
                            lines = nor_code_format.split('\n')
                            segment1, segment2 = segment_code(lines)
                            self.data.extend(segment1)
                            self.data.extend(segment2)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Xây dựng vocab sau khi load dữ liệu
        if not self.data:
            raise ValueError("No valid data found in directory. Please check the data directory.")
        self.tokenizer.build_vocab(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]
        encoded = self.tokenizer.encode(code)

        # Cắt nếu chuỗi dài hơn max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]

        # Chuyển thành tensor
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)  # Input: tất cả trừ token cuối
        target_ids = torch.tensor(encoded[1:], dtype=torch.long)  # Target: tất cả trừ token đầu

        return input_ids, target_ids
    
import re

def split_code_into_segments(code: str) -> list[str]:
    code = remove_comments(code)
    lines = code.strip().splitlines()
    result = []
    stack = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Bỏ dòng chỉ có dấu đóng block
        if stripped == "}":
            if stack:
                stack.pop()
            continue

        # Giữ nguyên annotation, comment
        if stripped.startswith("@") or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            result.append(stripped)
            continue

        # Nếu dòng kết thúc bằng {, thay { bằng {}
        if stripped.endswith("{"):
            keyword_line = stripped[:-1].strip()
            result.append(keyword_line + " {}")
            stack.append("{")
            continue

        # Tách các câu có dấu chấm phẩy
        if ";" in stripped and not stripped.startswith("case") and not stripped.startswith("default"):
            # Dùng regex để tách vừa dấu ; vừa dấu {, } và không mất các ký tự đó
            tokens = re.findall(r'[^{;}]+[;]?|[{}]', stripped)
            # tokens ví dụ: ['public String getDeptId() { return deptId;', '}', ';']

            # Dùng list tạm để gom token liên quan
            temp = ""
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue

                # Nếu token là dấu đóng block }
                if tok == "}":
                    if stack:
                        stack.pop()
                    # Bỏ token dấu } theo yêu cầu (không thêm vào result)
                    continue

                # Nếu token là dấu mở block {
                if tok == "{":
                    stack.append("{")
                    # Thay { thành {}
                    if temp:
                        result.append(temp + " {}")
                        temp = ""
                    else:
                        result.append("{}")
                    continue

                # Nếu token kết thúc bằng ; thì kết thúc 1 câu
                if tok.endswith(";"):
                    temp += tok
                    result.append(temp.strip())
                    temp = ""
                    continue

                # Các trường hợp còn lại nối vào temp
                temp += tok + " "

            # Nếu còn temp chưa được thêm
            if temp.strip():
                result.append(temp.strip())
            continue

        # Trường hợp có dấu } ở giữa dòng nhưng không phải dòng chỉ có }
        if "}" in stripped:
            # Thay thế } bằng dấu } và tách thành nhiều phần
            parts = re.split(r'(\})', stripped)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part == "}":
                    if stack:
                        stack.pop()
                    # Bỏ dấu } không thêm vào result
                    continue
                else:
                    result.append(part)
            continue

        # Các dòng còn lại thêm trực tiếp
        result.append(stripped)

    return result

def collate_fn(batch):
    """Padding các chuỗi trong batch để có cùng độ dài"""
    input_ids, target_ids = zip(*batch)
    # Padding các chuỗi
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # 0 là idx của <pad>
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    return input_ids, target_ids

def fix_malformed_comments(code: str) -> str:
    # Thay thế / / thành // nếu nó có vẻ là comment
    code = re.sub(r'(^|\s)/\s/+', r'\1//', code)
    return code

def remove_comments(code: list[str]) -> str:
    code_str = "\n".join(code)

    # Sửa các comment bị lỗi (dạng / / ...)
    code_str = fix_malformed_comments(code_str)

    # Xóa comment dạng /* ... */
    code_str = re.sub(r'/\*.*?\*/', '', code_str, flags=re.DOTALL)

    # Xóa comment dòng dạng //
    code_str = re.sub(r'//.*', '', code_str)

    return code_str

# *******
def segment_code(code):
    code = remove_comments(code)
    lines = code.split('\n')
    def extract_blocks(lines):
        blocks = []
        stack = []
        current_block = []
        for line in lines:
            stripped = line.strip()
            if '{' in stripped:
                if current_block:
                    stack.append(current_block)
                current_block = [line]
                if '}' in stripped and stripped.index('}') > stripped.index('{'):
                    one_line = ' '.join(l.strip() for l in current_block)
                    blocks.append(one_line)
                    current_block = stack.pop() if stack else []
            elif '}' in stripped:
                current_block.append(line)
                one_line = ' '.join(l.strip() for l in current_block)
                blocks.append(one_line)
                current_block = stack.pop() if stack else []
            else:
                current_block.append(line)

        if current_block and any("{" in line or "}" in line for line in current_block):
            one_line = ' '.join(l.strip() for l in current_block)
            blocks.append(one_line)

        return blocks

    # Segment 1: lấy block code
    segment1_blocks = ["<block> " + block for block in extract_blocks(lines)]

    segment2_lines = ["<line> " + line for line in split_code_into_segments(lines)]

    return segment1_blocks, segment2_lines


def extract_blocks_hierarchical(lines):
    """
    Phân tách code theo dấu { } với cấu trúc phân cấp:
    Mỗi block là một dict:
    {
        'header': dòng mở đầu (có dấu {),
        'body': list các dòng hoặc các block con (có thể là string hoặc dict)
    }
    """
    blocks_stack = []
    current_block = {'header': None, 'body': []}

    for line in lines:
        stripped = line.strip()

        # Nếu dòng có dấu { (mở block mới)
        if '{' in stripped:
            # Lấy phần trước dấu { làm header block
            header = stripped.split('{')[0].strip() + '{'
            # Tạo block mới
            new_block = {'header': header, 'body': []}

            # Nếu current_block đã có header thì push vào stack
            if current_block['header'] is not None:
                blocks_stack.append(current_block)

            # Chuyển current_block thành new_block
            current_block = new_block

            # Nếu dòng đó có cả dấu } sau dấu { thì block kết thúc ngay (ví dụ: for(...) {})
            if '}' in stripped and stripped.index('}') > stripped.index('{'):
                # Thêm block hiện tại vào block cha (nếu có) hoặc trả về
                if blocks_stack:
                    parent = blocks_stack.pop()
                    parent['body'].append(current_block)
                    current_block = parent
                else:
                    # Đây là block gốc
                    pass

        # Nếu dòng có dấu } (đóng block)
        elif '}' in stripped:
            # Kết thúc block hiện tại
            if blocks_stack:
                parent = blocks_stack.pop()
                parent['body'].append(current_block)
                current_block = parent
            else:
                # Đóng block gốc
                pass

        else:
            # Dòng bình thường thêm vào body của current block
            current_block['body'].append(line)

    return current_block

def block_to_string(block, indent=0):
    """
    Chuyển block dict thành string code có indent
    """
    space = '    ' * indent
    if isinstance(block, str):
        return space + block
    result = []
    if block['header']:
        result.append(space + block['header'])
    for b in block['body']:
        if isinstance(b, dict):
            result.append(block_to_string(b, indent + 1))
        else:
            # b là dòng string
            result.append(space + '    ' + b)
    result.append(space + '}')
    return '\n'.join(result)

def normalize_code_format(code: str) -> str:
    # Chuẩn hóa line endings và BOM
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = code.replace('\t', '    ')
    if code.startswith('\ufeff'):
        code = code[1:]

    # Xóa khoảng trắng cuối dòng
    code = '\n'.join(line.rstrip() for line in code.splitlines())

    # Thêm dấu cách sau từ khóa điều kiện
    code = re.sub(r'\b(for|if|while|switch|catch)\(', r'\1 (', code)
    code = re.sub(r'\b(for|if|while|switch|catch)\s+\(', r'\1 (', code)

    code = code.replace('++', '__PLUSPLUS__').replace('--', '__MINUSMINUS__')
    code = re.sub(r'(import\s+[a-zA-Z0-9_.]+)\s*\*', r'\1__STAR__', code)

    # Thêm dấu cách quanh các toán tử: = + - * / < > == != <= >=
    surround_operators = [
        r'==', r'!=', r'<=', r'>=', r'\+', r'-', r'/', r'=', r'<', r'>', r'\*'
    ]
    for op in surround_operators:
        code = re.sub(fr'\s*{op}\s*', f' {op.strip("\\")} ', code)

    postfix_operators = [';']

    for op in postfix_operators:
        code = re.sub(fr'{op}(?!\s)', f'{op} ', code)

    # Loại bỏ nhiều dấu cách liên tiếp (trừ trong string)
    code = re.sub(r'[ ]{2,}', ' ', code)

    code = code.replace('__PLUSPLUS__', '++').replace('__MINUSMINUS__', '--')
    code = code.replace('__STAR__', '*')

    return code

if __name__ == "__main__":
    data_dir = r"D:\Desktop\DACN\dacn_1\data\testdata"

    # Duyệt qua tất cả các file trong thư mục testdata
    for filename in os.listdir(data_dir):
        if filename.endswith(".java"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
                nor = normalize_code_format(code)
                print(nor)
        lines = nor.split('\n')
        segment1, segment2 = segment_code(lines)

        print("--- Segment 1 ---")
        for i, block in enumerate(segment1):
            print(f"Block {i+1}:\n{block}\n")

        print("--- Segment 2 ---")
        for line in segment2:
            print(line)