for DIR in figs/*; do
    if [[ ! -d $DIR ]]; then
        continue
    fi

    if [[ ! -d videos ]]; then
        mkdir videos
    fi

    seq=$(basename -- $DIR)

    ffmpeg -r 20 -pattern_type glob -i "${DIR}/im_raw_*.png" -c:v libx264 -vf fps=30 -pix_fmt yuv420p "videos/${seq}__im_raw.mp4"
    ffmpeg -r 20 -pattern_type glob -i "${DIR}/im_pl_*.png" -c:v libx264 -vf fps=30 -pix_fmt yuv420p "videos/${seq}__im_pl.mp4"
    ffmpeg -r 20 -pattern_type glob -i "${DIR}/scan_det_*.png" -c:v libx264 -vf fps=30 -pix_fmt yuv420p "videos/${seq}__det.mp4"
    ffmpeg -r 20 -pattern_type glob -i "${DIR}/scan_pl_*.png" -c:v libx264 -vf fps=30 -pix_fmt yuv420p "videos/${seq}__pl.mp4"
done
