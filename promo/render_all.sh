set -e
FPS=30; N=1935
for dim in "1920 1080 L" "1080 1920 P"; do
  set -- $dim
  K=4; step=$(( (N + K - 1) / K )); pids=""
  for i in $(seq 0 $((K-1))); do
    a=$((i*step)); b=$(( (i+1)*step )); [ $b -gt $N ] && b=$N
    node render.js $1 $2 $FPS $a $b seg_$3_$i.mp4 & pids="$pids $!"
  done
  wait $pids
  printf "file 'seg_$3_%d.mp4'\n" 0 1 2 3 > list_$3.txt
done
echo DONE
