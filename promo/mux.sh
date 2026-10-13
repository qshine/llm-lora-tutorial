# concat segments, add fine film grain, mux the score (loudness-normalised, faded out)
set -e
for X in L P; do
  out=$([ $X = L ] && echo lora-promo-16x9.mp4 || echo lora-promo-9x16.mp4)
  ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i list_$X.txt -i music.wav \
    -filter_complex "[0:v]noise=alls=2:allf=t,format=yuv420p[v];[1:a]atrim=0:64.5,afade=t=out:st=62.2:d=2.3,loudnorm=I=-16:TP=-1.5:LRA=11[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -preset slow -crf 20 -maxrate 9M -bufsize 18M -tune film -profile:v high -pix_fmt yuv420p -r 30 \
    -c:a aac -b:a 256k -ar 48000 -movflags +faststart -t 64.5 $out
  ffprobe -v error -show_entries format=duration,size -of csv=p=0 $out
done
