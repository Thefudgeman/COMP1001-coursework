#!/bin/bash
echo What is the file path of your picture?
read FilePath
gcc code_to_start/image_processing.c -o p -O3 -lm
./p "$FilePath"

