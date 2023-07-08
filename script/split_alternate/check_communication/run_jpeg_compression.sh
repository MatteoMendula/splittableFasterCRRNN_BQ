#!/bin/bash
for cmp in 10 20 25 30 40 70 100; do
    python3 check_communication_jpeg.py -c=${cmp}
done