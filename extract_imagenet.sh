#!/bin/bash
# ImageNet Dataset Extraction Script
# Extracts imagenet-val.tar.gz into proper class folder structure for PyTorch ImageFolder

set -e  # Exit on any error

# Configuration
SOURCE_TAR="~/BFL/imagenet-val.tar.gz"
OUTPUT_DIR="~/BFL/imagenet_extracted"
TEMP_DIR="~/BFL/temp_imagenet"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ImageNet Dataset Extraction Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Expand paths
SOURCE_TAR=$(eval echo $SOURCE_TAR)
OUTPUT_DIR=$(eval echo $OUTPUT_DIR)
TEMP_DIR=$(eval echo $TEMP_DIR)

echo -e "${YELLOW}Configuration:${NC}"
echo "  Source: $SOURCE_TAR"
echo "  Output: $OUTPUT_DIR"
echo "  Temp:   $TEMP_DIR"
echo ""

# Check if source file exists
if [ ! -f "$SOURCE_TAR" ]; then
    echo -e "${RED}❌ Error: Source file not found: $SOURCE_TAR${NC}"
    exit 1
fi

# Create directories
echo -e "${YELLOW}Step 1: Creating directories...${NC}"
mkdir -p "$OUTPUT_DIR/val"
mkdir -p "$TEMP_DIR"

# Extract main tar.gz
echo -e "${YELLOW}Step 2: Extracting main tar.gz file...${NC}"
cd "$(dirname "$SOURCE_TAR")"
tar -xzf "$SOURCE_TAR" -C "$TEMP_DIR"

# Find the extracted directory structure
IMAGENET_DIR=$(find "$TEMP_DIR" -name "validation" -type d | head -1)
if [ -z "$IMAGENET_DIR" ]; then
    IMAGENET_DIR=$(find "$TEMP_DIR" -name "imagenet" -type d | head -1)
fi

if [ -z "$IMAGENET_DIR" ]; then
    echo -e "${RED}❌ Error: Could not find validation or imagenet directory in extracted files${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found ImageNet directory: $IMAGENET_DIR${NC}"

# Extract nested tar files
echo -e "${YELLOW}Step 3: Extracting nested tar files...${NC}"
NESTED_TAR_DIR="$IMAGENET_DIR"
if [ -d "$IMAGENET_DIR/validation" ]; then
    NESTED_TAR_DIR="$IMAGENET_DIR/validation"
fi

# Find all .tar files in subdirectories
TAR_FILES=$(find "$NESTED_TAR_DIR" -name "*.tar" -type f)
TAR_COUNT=$(echo "$TAR_FILES" | wc -l)

echo "Found $TAR_COUNT nested tar files to extract"

# Extract each tar file
CURRENT=0
for tar_file in $TAR_FILES; do
    CURRENT=$((CURRENT + 1))
    echo -ne "${BLUE}Extracting tar file $CURRENT/$TAR_COUNT...${NC}\r"
    tar -xf "$tar_file" -C "$OUTPUT_DIR/val/" 2>/dev/null || true
done
echo -e "${GREEN}✅ Extracted all nested tar files${NC}"

# Organize images into class folders
echo -e "${YELLOW}Step 4: Organizing images into class folders...${NC}"
cd "$OUTPUT_DIR/val"

# Count total images
TOTAL_IMAGES=$(find . -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
echo "Found $TOTAL_IMAGES images to organize"

# Organize images by class
PROCESSED=0
for file in *.jpg *.jpeg *.png; do
    [ -f "$file" ] || continue
    
    # Extract class name from filename (format: ILSVRC2012_val_XXXXXXXX_nXXXXXXXX.jpg)
    class=$(echo "$file" | sed -E 's/.*_(n[0-9]+)\.(jpg|jpeg|png)$/\1/')
    
    if [[ "$class" =~ ^n[0-9]+$ ]]; then
        mkdir -p "$class"
        mv "$file" "$class/"
        PROCESSED=$((PROCESSED + 1))
        
        # Progress indicator
        if [ $((PROCESSED % 100)) -eq 0 ]; then
            echo -ne "${BLUE}Organized $PROCESSED/$TOTAL_IMAGES images...${NC}\r"
        fi
    fi
done
echo -e "${GREEN}✅ Organized $PROCESSED images into class folders${NC}"

# Clean up JSON files
echo -e "${YELLOW}Step 5: Cleaning up JSON files...${NC}"
find "$OUTPUT_DIR/val" -name "*.json" -delete 2>/dev/null || true

# Clean up temp directory
echo -e "${YELLOW}Step 6: Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ ImageNet extraction completed!${NC}"
echo -e "${GREEN}========================================${NC}"

# Count final results
CLASS_COUNT=$(find "$OUTPUT_DIR/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
IMAGE_COUNT=$(find "$OUTPUT_DIR/val" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)

echo -e "${BLUE}Final dataset statistics:${NC}"
echo "  Location: $OUTPUT_DIR/val"
echo "  Classes:  $CLASS_COUNT"
echo "  Images:   $IMAGE_COUNT"
echo ""
echo -e "${YELLOW}Usage in your script:${NC}"
echo "  python alignment_analysis.py --imagenet-path $OUTPUT_DIR --batch-size 2 --device cpu"
echo ""
echo -e "${GREEN}🎉 Ready to run FLUX alignment analysis!${NC}"