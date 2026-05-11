"""
Pipeline de Dados - TáFresco
Balanceia as classes (undersampling) e divide em train/val/test.
"""
import os
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── Configurações ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLASSES = ["Not_Fresh", "Fresh", "Highly_Fresh"]
LIMIT_PER_CLASS = 1760  # quantidade da classe minoritária
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def setup_folders():
    """Cria estrutura de pastas para os splits."""
    for split in SPLITS:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def copy_worker(task):
    """Copia um ficheiro (auxiliar para ThreadPoolExecutor)."""
    src, dst = task
    shutil.copy2(src, dst)


def balance_and_split():
    """Balanceia classes por undersampling e divide em train/val/test."""
    all_copy_tasks = []

    for cls in CLASSES:
        cls_path = RAW_DIR / cls
        # filtra imagens válidas
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # undersampling
        random.seed(42)
        selected_images = random.sample(images, min(len(images), LIMIT_PER_CLASS))
        random.shuffle(selected_images)

        # índices do split
        total = len(selected_images)
        idx_train = int(total * SPLITS["train"])
        idx_val = idx_train + int(total * SPLITS["val"])

        split_map = {
            "train": selected_images[:idx_train],
            "val": selected_images[idx_train:idx_val],
            "test": selected_images[idx_val:],
        }

        # preparação para o threadpool
        for split, imgs in split_map.items():
            for img in imgs:
                src = cls_path / img
                dst = PROCESSED_DIR / split / cls / img
                all_copy_tasks.append((src, dst))

        print(
            f"Classe {cls}: {len(selected_images)} imagens selecionadas. "
            f"→ train={idx_train}, val={idx_val - idx_train}, "
            f"test={total - idx_val}"
        )

    # execução paralela (fora do loop de classes)
    print(f"\nCopiando {len(all_copy_tasks)} ficheiros em paralelo...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(copy_worker, all_copy_tasks))


if __name__ == "__main__":
    setup_folders()
    balance_and_split()
    print(f"\n✅ Dataset pronto em: {PROCESSED_DIR}")
