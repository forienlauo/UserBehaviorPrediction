#!/usr/bin/env bash

set -u
set +e

exp_cnt=5000
ccs=64,128,256
l=20
p=0.8
b=40
i=500
k=0.8
r=0.00000001
G=0
M=0.3
S=1
prefix=cm
suffix=ccs$ccs-l$l-c$exp_cnt-p$p-b$b-k$k-i$i-r$r
log_dir=tmp/$prefix-log-$suffix
train_dir=tmp/$prefix-train_wkdir-$suffix
cache_dir=tmp/sampled_cache-$exp_cnt
mkdir $log_dir
python cm_train.py -f null -t null -c $cache_dir -p $p -w $train_dir -ccs $ccs -l $l -b $b -k $k -i $i -r $r -G $G -M $M -S $S -F > $log_dir/`date +'%Y%m%d-%H%M%S'`.log 2>&1
