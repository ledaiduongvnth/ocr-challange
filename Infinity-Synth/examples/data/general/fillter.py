# import json
# import os

# # Đường dẫn file input
# input_file = "/home/khaint02/Desktop/ocr-challange/output_folder_ADE/1. BIEU PHI DICH VU TAI KHOAN_page_0001_parse_output.json"  # Thay đường dẫn file input của bạn
# output_text_file = "/home/khaint02/Desktop/ocr-challange/Infinity-Synth/examples/data/general/text.json"
# output_table_file = "/home/khaint02/Desktop/ocr-challange/Infinity-Synth/examples/data/general/table.json"

# # Đọc file input
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Khởi tạo danh sách cho text và table
# text_list = []
# table_list = []

# # Lọc chunks theo type
# if "chunks" in data:
#     for chunk in data["chunks"]:
#         chunk_type = chunk.get("type", "").lower()
#         markdown = chunk.get("markdown", "")
        
#         if chunk_type == "text" and markdown:
#             text_list.append({
#                 "type": "text",
#                 "content": markdown
#             })
#         elif chunk_type == "table" and markdown:
#             table_list.append({
#                 "type": "table",
#                 "content": markdown
#             })

# # Ghi vào file text.json
# if text_list:
#     with open(output_text_file, 'w', encoding='utf-8') as f:
#         json.dump(text_list, f, ensure_ascii=False, indent=2)
#     print(f"✅ Đã ghi {len(text_list)} text chunks vào {output_text_file}")

# # Ghi vào file table.json
# if table_list:
#     with open(output_table_file, 'w', encoding='utf-8') as f:
#         json.dump(table_list, f, ensure_ascii=False, indent=2)
#     print(f"✅ Đã ghi {len(table_list)} table chunks vào {output_table_file}")

# if not text_list and not table_list:
#     print("⚠️  Không tìm thấy chunks nào có type 'text' hoặc 'table'")

import json
import os
from pathlib import Path

# Đường dẫn folder input và output
input_folder = "/home/khaint02/Desktop/ocr-challange/output_folder_ADE"
output_base_folder = "/home/khaint02/Desktop/ocr-challange/Infinity-Synth/examples/data/general"

def process_folder(folder_path):
    """Xử lý tất cả file JSON trong folder"""
    
    # Duyệt tất cả file JSON trong folder
    json_files = list(Path(folder_path).glob("**/*.json"))
    
    if not json_files:
        print(f"⚠️  Không tìm thấy file JSON nào trong {folder_path}")
        return
    
    print(f"📁 Tìm thấy {len(json_files)} file JSON")
    
    # Khởi tạo danh sách chung cho tất cả file
    all_text_list = []
    all_table_list = []
    
    for json_file in json_files:
        print(f"\n🔄 Đang xử lý: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Lọc chunks theo type
            if isinstance(data, dict) and "chunks" in data:
                chunks = data.get("chunks", [])
                
                for chunk in chunks:
                    chunk_type = chunk.get("type", "").lower().strip()
                    markdown = chunk.get("markdown", "")
                    
                    # Lọc type "text"
                    if chunk_type == "text" and markdown:
                        all_text_list.append({
                            "type": "text",
                            "content": markdown.strip()
                        })
                    
                    # Lọc type "table"
                    elif chunk_type == "table" and markdown:
                        all_table_list.append({
                            "type": "table",
                            "content": markdown.strip()
                        })
                
                print(f"   ✅ Tìm thấy {len([c for c in chunks if c.get('type', '').lower().strip() == 'text'])} text chunks")
                print(f"   ✅ Tìm thấy {len([c for c in chunks if c.get('type', '').lower().strip() == 'table'])} table chunks")
        
        except json.JSONDecodeError as e:
            print(f"   ❌ Lỗi decode JSON: {e}")
        except Exception as e:
            print(f"   ❌ Lỗi xử lý: {e}")
    
    # Tạo folder output nếu chưa tồn tại
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Ghi text.json chung
    if all_text_list:
        output_text_file = os.path.join(output_base_folder, "text.json")
        with open(output_text_file, 'w', encoding='utf-8') as f:
            json.dump(all_text_list, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Ghi {len(all_text_list)} text chunks → {output_text_file}")
    
    # Ghi table.json chung
    if all_table_list:
        output_table_file = os.path.join(output_base_folder, "table.json")
        with open(output_table_file, 'w', encoding='utf-8') as f:
            json.dump(all_table_list, f, ensure_ascii=False, indent=2)
        print(f"✅ Ghi {len(all_table_list)} table chunks → {output_table_file}")
    
    if not all_text_list and not all_table_list:
        print(f"\n⚠️  Không tìm thấy text hoặc table chunks")

# Chạy script
if __name__ == "__main__":
    print("🚀 Bắt đầu xử lý folder...")
    process_folder(input_folder)
    print("\n✨ Hoàn thành!")