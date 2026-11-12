import sys
 
from app.config import get_settings
from app.rag import build_rag_service

def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python -m scripts.ask "sua pergunta"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    service = build_rag_service(get_settings())
    result = service.ask(question)

    print(f"\nPergunta: {question}\n")
    print(result.answer)
    print("\nFontes:")
    for s in result.sources:
        print(f"  - {s['arquivo']}, p. {s['pagina']}")

if __name__ == "__main__":
    main()