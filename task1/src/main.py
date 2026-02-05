import sys
import re
from pathlib import Path
from rapidfuzz import process, fuzz

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"

def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def find_matches(query: str, names: list[str], limit: int = 10) -> list[tuple[str, float]]:
    results = process.extract(query, names, scorer=fuzz.WRatio, limit=limit)
    # results: list of (name, score, index)
    return [(name, float(score)) for name, score, _ in results]


def parse_top_n(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise ValueError("Top N must be an integer.") from None
    if n <= 0:
        raise ValueError("Top N must be a positive integer.")
    return n


def sanitize_name(raw: str) -> str:
    # Trim and collapse multiple spaces to a single space.
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return ""
    # Allow only alphabets and spaces.
    if not re.fullmatch(r"[A-Za-z ]+", cleaned):
        return ""
    return cleaned


def main() -> None:
    if len(sys.argv) < 2:
        query = input("Enter a name to match: ")
        top_n_raw = input("Enter Top N (default 10): ").strip()
        if top_n_raw:
            try:
                top_n = parse_top_n(top_n_raw)
            except ValueError as exc:
                print(str(exc))
                sys.exit(1)
        else:
            top_n = 10
    else:
        query = sys.argv[1]
        if len(sys.argv) >= 3:
            try:
                top_n = parse_top_n(sys.argv[2])
            except ValueError as exc:
                print(str(exc))
                sys.exit(1)
        else:
            top_n = 10

    query = sanitize_name(query)
    if not query:
        print("Please enter a valid name (alphabets and spaces only).")
        sys.exit(1)

    names = load_names(DATA_PATH)
    matches = find_matches(query, names, limit=top_n)

    if not matches:
        print("No matches found.")
        return

    best_name, best_score = matches[0]
    print(f"Best Match: {best_name} (score: {best_score:.2f})")
    print("List of Matches:")
    for name, score in matches:
        print(f"- {name}: {score:.2f}")


if __name__ == "__main__":
    main()
