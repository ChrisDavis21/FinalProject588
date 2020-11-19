#!/bin/bash

#Threshold for histogram compare mask detection
thresh=0.45
maskCorrect=0
maskWrong=0
noMaskCorrect=0
noMaskWrong=0
for i in {1..40}
do
    val=$(python ./checkCluster.py data/raw/mask$i.jpg)
    printf "mask$i.jpg $val\n"
#    if awk 'BEGIN {exit !('$val' >= '$thresh')}'; then
#        echo "not detected"
#        maskWrong=$((maskWrong + 1))
#    else 
#        echo "mask found"
#        maskCorrect=$((maskCorrect + 1))
#    fi
done

for i in {1..40}
do
    val=$(python ./checkCluster.py data/raw/nomask$i.jpg)
    printf "nomask$i.jpg $val\n"
#    if awk 'BEGIN {exit !('$val' >= '$thresh')}'; then
#        echo "not detected"
#        noMaskCorrect=$((noMaskCorrect + 1))
#    else 
#        echo "mask found"
#        noMaskWrong=$((noMaskWrong + 1))
#    fi
done

#success=$((noMaskCorrect + maskCorrect))
#success=$((success * 100))
#success=$((success / 80))
#printf "\nHISTOGRAM COMPARISON RESULTS\n"
#printf "Correctly identified %d of 40 mask photos\n" $maskCorrect
#printf "Correctly identified %d of 40 no mask photos\n" $noMaskCorrect
#printf "Overall success rate on sample dataset: %d%%\n" $success