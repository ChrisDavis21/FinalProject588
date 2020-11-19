#!/bin/bash

#Threshold for histogram compare mask detection
thresh=0.65
maskCorrect=0
maskWrong=0
noMaskCorrect=0
noMaskWrong=0
for i in {1..40}
do
    val=$(python ./checkCluster.py data/raw/mask$i.jpg)
    printf "mask$i.jpg "
    if awk 'BEGIN {exit !('$val' <= '$thresh')}'; then
        echo "not detected $val"
        maskWrong=$((maskWrong + 1))
    else 
        echo "mask found $val"
        maskCorrect=$((maskCorrect + 1))
    fi
done

for i in {1..40}
do
    val=$(python ./checkCluster.py data/raw/nomask$i.jpg)
    printf "nomask$i.jpg "
    if awk 'BEGIN {exit !('$val' <= '$thresh')}'; then
        echo "not detected $val"
        noMaskCorrect=$((noMaskCorrect + 1))
    else 
        echo "mask found $val"
        noMaskWrong=$((noMaskWrong + 1))
    fi
done

success=$((noMaskCorrect + maskCorrect))
success=$((success * 100))
success=$((success / 80))
printf "\nCLUSTERING COMPARISON RESULTS\n"
printf "Correctly identified %d of 40 mask photos\n" $maskCorrect
printf "Correctly identified %d of 40 no mask photos\n" $noMaskCorrect
printf "Overall success rate on sample dataset: %d%%\n" $success