#!/bin/bash

ZIP_NAME="410410020_proj6.zip"

INCLUDE_ITEMS=(
    "train.py"
    "test.py"
    "models/"
    "datasets/"
#    "plant-seedlings-classification/"
#    "weights/"
    "result/"
    "predictions.csv"
    "410410020_proj6.pdf"
)

# 如果已存在相同 zip，先刪除
if [ -f "$ZIP_NAME" ]; then
    echo "❗ 已存在 $ZIP_NAME，將重新壓縮..."
    rm "$ZIP_NAME"
fi

# 執行壓縮
echo "壓縮中..."
zip -9 -r "$ZIP_NAME" "${INCLUDE_ITEMS[@]}"

echo "壓縮完成，已生成：$ZIP_NAME"