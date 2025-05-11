import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.tokenizer.tokenizer import Tokenizer

class JavaCodeDataset(Dataset):
    def __init__(self, directory, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load dữ liệu từ thư mục
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                            segments = self.split_code_into_segments(code)
                            self.data.extend(segments)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Xây dựng vocab sau khi load dữ liệu
        if not self.data:
            raise ValueError("No valid data found in directory. Please check the data directory.")
        self.tokenizer.build_vocab(self.data)

    # def split_code_into_segments(self, code: str) -> list[str]:
    #     """Chia code thành các đoạn nhỏ dựa trên dấu ;, khối {}, và cặp ()"""
    #     code = self.tokenizer.remove_comments(code)
    #     segments = []
    #     current_segment = ""
    #     brace_count = 0  # Đếm số { }
    #     paren_count = 0  # Đếm số ( )
    #     i = 0

    #     while i < len(code):
    #         char = code[i]
    #         current_segment += char

    #         if char == '(':
    #             paren_count += 1
    #             if paren_count == 1:  # Bắt đầu một cặp ()
    #                 start_segment = current_segment  # Lưu đoạn trước (
    #                 inner_segment = ""
    #                 i += 1
    #                 # Gộp toàn bộ nội dung trong cặp () thành một đoạn
    #                 while i < len(code) and paren_count > 0:
    #                     char = code[i]
    #                     inner_segment += char
    #                     if char == '(':
    #                         paren_count += 1
    #                     elif char == ')':
    #                         paren_count -= 1
    #                     i += 1
    #                 # Gộp đoạn trong () thành một đoạn duy nhất
    #                 combined_segment = start_segment + inner_segment
    #                 # Kiểm tra xem sau ) có { không
    #                 j = i
    #                 while j < len(code) and code[j].isspace():
    #                     j += 1
    #                 if j < len(code) and code[j] == '{':
    #                     # Thêm {} ngay sau ) để tạo thành đoạn
    #                     combined_segment += " {}"
    #                     segments.append(combined_segment.strip())  # Loại bỏ \n và khoảng trắng
    #                     current_segment = ""
    #                     # Bắt đầu xử lý khối {}
    #                     brace_count += 1
    #                     i = j + 1  # Bỏ qua {
    #                     # Tách các đoạn bên trong khối
    #                     inner_segments = []
    #                     inner_segment = ""
    #                     while i < len(code) and brace_count > 0:
    #                         char = code[i]
    #                         if char == '(':
    #                             paren_count += 1
    #                             inner_segment += char
    #                             i += 1
    #                             # Gộp nội dung trong cặp () trong khối
    #                             while i < len(code) and paren_count > 0:
    #                                 char = code[i]
    #                                 inner_segment += char
    #                                 if char == '(':
    #                                     paren_count += 1
    #                                 elif char == ')':
    #                                     paren_count -= 1
    #                                 i += 1
    #                             continue
    #                         elif char == '{':
    #                             brace_count += 1
    #                             # Không thêm {} nữa, vì đã thêm ở đoạn khai báo
    #                             inner_segment += char
    #                         elif char == '}':
    #                             brace_count -= 1
    #                             if brace_count > 0:  # Nếu không phải } cuối cùng, thêm vào inner_segment
    #                                 inner_segment += char
    #                             else:
    #                                 # Kết thúc khối, không thêm } vào đoạn
    #                                 if inner_segment:
    #                                     inner_segments.append(inner_segment.strip())
    #                                 inner_segment = ""
    #                         elif char == ';' and inner_segment:
    #                             inner_segments.append((inner_segment + ";").strip())
    #                             inner_segment = ""
    #                         else:
    #                             inner_segment += char
    #                         i += 1
    #                     # Thêm các đoạn trong khối vào segments
    #                     segments.extend(inner_segments)
    #                     continue
    #                 else:
    #                     # Nếu không có { sau ), chỉ thêm đoạn (....)
    #                     segments.append(combined_segment.strip())
    #                     current_segment = ""
    #                 continue
    #         elif char == '{':
    #             brace_count += 1
    #             if brace_count == 1:  # Bắt đầu một khối
    #                 # Thêm {} vào đoạn hiện tại
    #                 current_segment += " }"
    #                 segments.append(current_segment.strip())
    #                 current_segment = ""
    #                 i += 1
    #                 # Tách các đoạn bên trong khối
    #                 inner_segments = []
    #                 inner_segment = ""
    #                 while i < len(code) and brace_count > 0:
    #                     char = code[i]
    #                     if char == '(':
    #                         paren_count += 1
    #                         inner_segment += char
    #                         i += 1
    #                         # Gộp nội dung trong cặp () trong khối
    #                         while i < len(code) and paren_count > 0:
    #                             char = code[i]
    #                             inner_segment += char
    #                             if char == '(':
    #                                 paren_count += 1
    #                             elif char == ')':
    #                                 paren_count -= 1
    #                             i += 1
    #                         continue
    #                     elif char == '{':
    #                         brace_count += 1
    #                         # Không thêm {} nữa, vì đã thêm ở đoạn khai báo
    #                         inner_segment += char
    #                     elif char == '}':
    #                         brace_count -= 1
    #                         if brace_count > 0:  # Nếu không phải } cuối cùng, thêm vào inner_segment
    #                             inner_segment += char
    #                         else:
    #                             # Kết thúc khối, không thêm } vào đoạn
    #                             if inner_segment:
    #                                 inner_segments.append(inner_segment.strip())
    #                             inner_segment = ""
    #                     elif char == ';' and inner_segment:
    #                         inner_segments.append((inner_segment + ";").strip())
    #                         inner_segment = ""
    #                     else:
    #                         inner_segment += char
    #                     i += 1
    #                 # Thêm các đoạn trong khối vào segments
    #                 segments.extend(inner_segments)
    #                 continue
    #         elif char == ';' and brace_count == 0 and paren_count == 0 and current_segment:
    #             segments.append(current_segment.strip())
    #             current_segment = ""

    #         i += 1

    #     if current_segment:
    #         segments.append(current_segment.strip())

    #     # Loại bỏ các đoạn rỗng
    #     segments = [seg for seg in segments if seg]
    #     print(segments)
    #     return segments

    def split_code_into_segments(self, code: str) -> list[str]:
        code = self.tokenizer.remove_comments(code)
        lines = code.strip().splitlines()
        result = []
        stack = []

        for line in lines:
            stripped = line.strip()

            # Bỏ dòng trống
            if not stripped:
                continue

            # Bỏ dòng đóng khối
            if stripped == "}":
                if stack:
                    stack.pop()
                continue

            # Annotation: giữ nguyên
            if stripped.startswith("@"):
                result.append(stripped)
                continue

            # Comment: giữ lại nếu muốn model học style
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                result.append(stripped)
                continue

            # switch, if, else, for, while, catch, etc.
            if stripped.endswith("{"):
                keyword_line = stripped[:-1].strip()
                result.append(keyword_line + " {}")
                stack.append("{")
                continue

            # Multiline code có ; ở giữa → tách
            if ";" in stripped and not stripped.startswith("case") and not stripped.startswith("default"):
                parts = [part.strip() + ";" for part in stripped.split(";") if part.strip()]
                result.extend(parts)
                continue

            # case / default
            if stripped.startswith("case ") or stripped.startswith("default:"):
                result.append(stripped)
                continue

            # Câu kết thúc bằng }
            if "}" in stripped and stripped != "}":
                cleaned = stripped.replace("}", "").strip()
                if cleaned:
                    result.append(cleaned)
                continue

            # Dòng còn lại
            result.append(stripped)
        return result

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

def collate_fn(batch):
    """Padding các chuỗi trong batch để có cùng độ dài"""
    input_ids, target_ids = zip(*batch)
    # Padding các chuỗi
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # 0 là idx của <pad>
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    return input_ids, target_ids