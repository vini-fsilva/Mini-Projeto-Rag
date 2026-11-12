from app.config import get_settings
from app.ingest import build_index

def main() -> None:
    settings = get_settings()
    print(f"Lendo PDFs de '{settings.pdf_dir}'...")
    total = build_index(settings)
    print(f"{total} chunks indexados em '{settings.chroma_dir}'.")

if __name__ == "__main__":
    main()