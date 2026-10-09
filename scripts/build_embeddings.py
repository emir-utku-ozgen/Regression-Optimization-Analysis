"""
data/raw altındaki CSV'lerden (train.csv, test.csv, yeni_sorular.csv) embedding'leri
yeniden üretip data/processed altına .npy olarak yazar.

- Model: ytu-ce-cosmos/turkish-e5-large (1024 boyut, L2 normalize)
- E5 formatı: sorular "query: ", cevaplar "passage: " önekiyle kodlanır
- Bölme soru bazında yapılır: aynı sorunun doğru ve yanlış cevabı hep aynı
  tarafta kalır, böylece test setine soru sızmaz.

Kullanım:
    python scripts/build_embeddings.py              # embedding üret ve yaz
    python scripts/build_embeddings.py --dry-run    # sadece doğrula ve bölmeyi göster
"""
import argparse
import csv
import random
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

RAW_FILES = ['train.csv', 'test.csv', 'yeni_sorular.csv']
QUERY_PREFIX = 'query: '
PASSAGE_PREFIX = 'passage: '


def load_rows(raw_dir):
    rows = []
    for name in RAW_FILES:
        path = Path(raw_dir) / name
        n = 0
        with open(path, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader, start=2):
                soru, cevap, etiket = r['Soru'].strip(), r['Cevap'].strip(), r['Etiket'].strip()
                if etiket not in ('1', '-1'):
                    raise ValueError(f"{name}:{i} geçersiz etiket: {etiket!r}")
                rows.append((soru, cevap, int(etiket)))
                n += 1
        print(f"{name}: {n} satır")
    return rows


def split_by_question(rows, test_size, seed):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(row[0], []).append(row)

    # Her soruda bir doğru ve bir yanlış cevap beklenir; olmayanları sadece raporla
    for soru, g in groups.items():
        if sorted(r[2] for r in g) != [-1, 1]:
            print(f"  UYARI: beklenmeyen etiket dağılımı {[r[2] for r in g]} -> {soru!r}")

    questions = list(groups)
    random.Random(seed).shuffle(questions)
    n_test = round(len(questions) * test_size)
    test_q, train_q = questions[:n_test], questions[n_test:]

    train = [r for q in train_q for r in groups[q]]
    test = [r for q in test_q for r in groups[q]]
    return train, test, len(train_q), len(test_q)


def encode(model, texts, prefix, batch_size):
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size,
                       normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--out-dir', default='data/processed')
    ap.add_argument('--model', default='ytu-ce-cosmos/turkish-e5-large')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--dry-run', action='store_true',
                    help='Model yüklemeden veriyi doğrula ve bölmeyi yazdır')
    args = ap.parse_args()

    rows = load_rows(args.raw_dir)
    train, test, n_train_q, n_test_q = split_by_question(rows, args.test_size, args.seed)
    for name, part, nq in [('train', train, n_train_q), ('test', test, n_test_q)]:
        dist = Counter(r[2] for r in part)
        print(f"{name}: {nq} soru, {len(part)} satır, etiket dağılımı {{+1: {dist[1]}, -1: {dist[-1]}}}")

    if args.dry_run:
        print("Dry-run: dosya yazılmadı.")
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [('train', train), ('test', test)]:
        sorular = [r[0] for r in part]
        cevaplar = [r[1] for r in part]
        labels = np.array([r[2] for r in part], dtype=np.float64).reshape(-1, 1)

        np.save(out / f'{name}_query_embeddings.npy', encode(model, sorular, QUERY_PREFIX, args.batch_size))
        np.save(out / f'{name}_passage_embeddings.npy', encode(model, cevaplar, PASSAGE_PREFIX, args.batch_size))
        np.save(out / f'{name}_labels.npy', labels)
        print(f"{name}: {out}/{name}_*.npy yazıldı ({len(part)} satır)")


if __name__ == '__main__':
    main()
