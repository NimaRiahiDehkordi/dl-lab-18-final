#!/bin/sh
END=10

for ((i=1;i<=END;i++)); do
    zip -r $i.zip ./intensity/checkpoints/$i/*.txt
done

for ((i=1;i<=END;i++)); do
    zip -r $i.zip ./spacial/checkpoints/$i/*.txt
done

