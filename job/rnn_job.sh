#!/usr/bin/env bash

set -u
set +e

exp_cnt=5000
l=20
p=0.8
b=40
I=500
k=0.8
r=0.00000001
G=0
M=0.5
S=1
prefix=rnn
suffix=l$l-c$exp_cnt-p$p-b$b-k$k-I$I-r$r
log_dir=tmp/$prefix-log-$suffix
train_dir=tmp/$prefix-train_wkdir-$suffix
cache_dir=tmp/sampled_cache-$exp_cnt
mkdir $log_dir
python rnn_train.py -f null -t null -c $cache_dir -p $p -w $train_dir -l $l -b $b -k $k -r $r -G $G -M $M -I $I -S $S -F > $log_dir/`date +'%Y%m%d-%H%M%S'`.log 2>&1
