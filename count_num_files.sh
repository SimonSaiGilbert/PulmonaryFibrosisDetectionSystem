#!/bin/sh

TARGET_DIR=$1
MAX_NUM_FILE=0

# find $TARGET_DIR -maxdepth 1 -type d -exec echo {} $(ls -1 {}) \;

for file in $(find $TARGET_DIR -maxdepth 1); do
    if [ -d $file ]; then
        NUM_FILE=$(ls -1 $file | wc -l)

        if [ $NUM_FILE -gt $MAX_NUM_FILE ]; then
            MAX_NUM_FILE=$NUM_FILE
            echo "$file: $NUM_FILE / $MAX_NUM_FILE" 
        fi
    fi
done

