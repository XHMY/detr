#!/bin/bash

# Set the base directory where COCO will be downloaded
base_dir="/scratch/open_coco"

# Create the base directory if it doesn't exist
mkdir -p "$base_dir"

# Change to the base directory
cd "$base_dir"

# Download the dataset
echo "Downloading COCO 2017 dataset..."
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract the dataset
echo "Extracting COCO 2017 dataset..."
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Move the annotations to the desired directory
mv annotations_trainval2017/annotations .

# Remove the zip files and unnecessary directory
echo "Cleaning up..."
rm train2017.zip val2017.zip annotations_trainval2017.zip
rm -rf annotations_trainval2017

echo "COCO 2017 dataset downloaded and extracted successfully!"

chmod 755 -R $base_dir