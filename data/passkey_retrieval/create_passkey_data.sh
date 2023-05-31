#!/bin/bash

echo dumping to $1

base_command="python data/passkey_retrieval/create_passkey_data.py --token-length"

start_text_len=2048
end_text_len=16384
step=2048

for ((text_len=$start_text_len; text_len<=$end_text_len; text_len+=step))
do
    export TEXT_LEN=$text_len

    full_command="$base_command $TEXT_LEN --dump-file-path $1/${TEXT_LEN} --tokenizer-path $2"

    echo "Creating retrieve1st benchmark with TEXT_LEN=$text_len"
    PYTHONPATH=. $full_command

    echo
done
