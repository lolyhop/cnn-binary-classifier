#!/bin/bash
echo "Downloading cats vs dogs dataset..."

mkdir -p data/raw
curl -L -o data/raw/dataset.zip \
  "https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset"

cd data/raw
unzip -q dataset.zip
cd ../..

echo "Dataset downloaded and extracted to data/raw/"
