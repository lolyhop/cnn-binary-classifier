from datetime import datetime
from pathlib import Path
import typing as tp
import shutil
import random
from PIL import Image
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    "etl_pipeline",
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

BASE_PATH = Path("/opt/airflow/data")
RAW_PATH = BASE_PATH / "raw" / "PetImages"
PROCESSED_PATH = BASE_PATH / "processed"


def extract_images(**context) -> tp.Dict[str, tp.List[str]]:
    """Load all images from Cat and Dog folders"""
    cats = [str(p) for p in (RAW_PATH / "Cat").glob("*.jpg")]
    dogs = [str(p) for p in (RAW_PATH / "Dog").glob("*.jpg")]

    result = {"cats": cats, "dogs": dogs}
    print(f"Loaded: {len(cats)} cats, {len(dogs)} dogs")
    return result


def clean_images(**context) -> tp.Dict[str, tp.Any]:
    """Clean data by removing corrupted images"""
    data = context["task_instance"].xcom_pull(task_ids="extract")

    def is_valid_image(img_path: str) -> bool:
        """Check if image can be verified"""
        try:
            with Image.open(Path(img_path)) as img:
                img.verify()
            return True
        except Exception:
            return False

    # Clean cats and dogs
    clean_cats = [img for img in data["cats"] if is_valid_image(img)]
    clean_dogs = [img for img in data["dogs"] if is_valid_image(img)]

    result = {
        "clean_cats": clean_cats,
        "clean_dogs": clean_dogs,
        "cats_removed": len(data["cats"]) - len(clean_cats),
        "dogs_removed": len(data["dogs"]) - len(clean_dogs),
    }

    print(f"Cleaning results:")
    print(
        f"  Cats: {len(data['cats'])} → {len(clean_cats)} (removed {result['cats_removed']})"
    )
    print(
        f"  Dogs: {len(data['dogs'])} → {len(clean_dogs)} (removed {result['dogs_removed']})"
    )

    return result


def split_dataset(**context) -> str:
    """Split cleaned data into train/val/test (70/20/10) and save"""
    data = context["task_instance"].xcom_pull(task_ids="clean")

    # Create directory structure
    for split in ["train", "val", "test"]:
        for category in ["cats", "dogs"]:
            (PROCESSED_PATH / split / category).mkdir(parents=True, exist_ok=True)

    # Convert to Path objects and shuffle
    cats = [Path(p) for p in data["clean_cats"]]
    dogs = [Path(p) for p in data["clean_dogs"]]
    random.shuffle(cats)
    random.shuffle(dogs)

    def split_data(
        images: tp.List[Path],
    ) -> tp.Tuple[tp.List[Path], tp.List[Path], tp.List[Path]]:
        """Split images into train/val/test (70/20/10)"""
        total = len(images)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)

        return (images[:train_end], images[train_end:val_end], images[val_end:])

    cats_train, cats_val, cats_test = split_data(cats)
    dogs_train, dogs_val, dogs_test = split_data(dogs)

    # Copy files to respective directories
    splits = [
        (cats_train, "train", "cats"),
        (cats_val, "val", "cats"),
        (cats_test, "test", "cats"),
        (dogs_train, "train", "dogs"),
        (dogs_val, "val", "dogs"),
        (dogs_test, "test", "dogs"),
    ]

    for images, split, category in splits:
        for img in images:
            shutil.copy2(img, PROCESSED_PATH / split / category / img.name)

    stats = {
        "train": len(cats_train) + len(dogs_train),
        "val": len(cats_val) + len(dogs_val),
        "test": len(cats_test) + len(dogs_test),
    }

    result = f"Dataset split complete - Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}"
    return result


extract_task = PythonOperator(
    task_id="extract", python_callable=extract_images, dag=dag
)
clean_task = PythonOperator(task_id="clean", python_callable=clean_images, dag=dag)
split_task = PythonOperator(task_id="split", python_callable=split_dataset, dag=dag)

extract_task >> clean_task >> split_task
