import logging
import sys
import time
import argparse
from pathlib import Path
import json
from enum import Enum # THÊM DÒNG NÀY

# Giả định rằng các module này tồn tại trong cấu trúc project của bạn
from src_bot.analyzers.analyzer_factory import AnalyzerFactory
from src_bot.neo4jdb.neo4j_service import Neo4jService
from src_bot.neo4jdb.neo4j_db import Neo4jDB

from src_bot.config.config import configs

# --- THÊM CLASS NÀY ĐỂ SỬA LỖI ---
class CustomJsonEncoder(json.JSONEncoder):
    """
    Bộ mã hóa JSON tùy chỉnh để xử lý các kiểu dữ liệu không chuẩn,
    cụ thể là các đối tượng Enum.
    """
    def default(self, obj):
        if isinstance(obj, Enum):
            # Nếu đối tượng là một Enum, trả về giá trị (value) của nó
            return obj.value
        # Đối với các kiểu khác, dùng cách xử lý mặc định
        return json.JSONEncoder.default(self, obj)
# ------------------------------------

def main():
    """
    Hàm chính để chạy quá trình phân tích source code,
    xuất dữ liệu ra JSON và nhập vào Neo4j.
    """
    start_time = time.perf_counter()

    # --- Cấu hình Parser cho tham số dòng lệnh ---
    parser = argparse.ArgumentParser(description="Phân tích source code và nhập vào cơ sở dữ liệu.")
    parser.add_argument("--project-path", required=True, help="Đường dẫn đến thư mục project cần phân tích.")
    parser.add_argument("--output-file", required=True, help="Đường dẫn đến file JSON để lưu kết quả phân tích.")
    parser.add_argument("--language", default="java", help="Ngôn ngữ lập trình của project (mặc định: java).")
    parser.add_argument("--project-id", default="default_project", help="ID định danh cho project.")
    args = parser.parse_args()

    # --- Cấu hình Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('analyzer.log'),
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # --- 1. Phân tích Source Code ---
        logger.info(f"Bắt đầu phân tích project tại: {args.project_path}")
        project_path = Path(args.project_path)
        
        analyzer = AnalyzerFactory.create_analyzer(args.language, str(project_path), args.project_id, "main")
        
        with analyzer as a:
            chunks = a.parse_project(project_path)

        logger.info(f"Phân tích hoàn tất. Tìm thấy {len(chunks)} code chunks (classes/interfaces/enums).")

        # --- 2. Xuất kết quả ra file JSON ---
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_for_json = [chunk.to_dict() for chunk in chunks]
        
        # SỬA DÒNG NÀY:
        # Thêm tham số `cls=CustomJsonEncoder` để sử dụng bộ mã hóa tùy chỉnh
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_for_json, f, indent=4, ensure_ascii=False, cls=CustomJsonEncoder)
        
        logger.info(f"Đã xuất dữ liệu phân tích ra file: {output_file}")

        # --- 3. Nhập dữ liệu vào Neo4j ---
        logger.info("Đang nhập dữ liệu vào Neo4j...")
        
        db = Neo4jDB(
            url=configs.APP_NEO4J_URL,
            user=configs.APP_NEO4J_USER,
            password=configs.APP_NEO4J_PASSWORD
        )
        neo4j_service = Neo4jService(db=db)
        
        import_start = time.perf_counter()
        neo4j_service.import_code_chunks_simple(
            chunks=chunks,
            batch_size=500
        )
        import_elapsed = time.perf_counter() - import_start
        logger.info(f"Đã nhập {len(chunks)} chunks vào Neo4j trong {import_elapsed:.2f} giây.")

        elapsed = time.perf_counter() - start_time
        logger.info(f"\nHoàn tất toàn bộ quá trình trong {elapsed:.2f} giây!")
        return 0

    except Exception as e:
        logger.error(f"Lỗi trong quá trình phân tích: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())