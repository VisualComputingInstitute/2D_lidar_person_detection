import pickle
import numpy as np

p1 = "/home/jia/git/awesome_repos/2D_lidar_person_detection/dr_spaam/logs/20210520_224024_drow_jrdb_EVAL/output/val/e000000/evaluation/all/result_r05.pkl"
p2 = "/home/jia/git/awesome_repos/2D_lidar_person_detection/dr_spaam/logs/20210520_231344_drow_jrdb_EVAL/output/val/e000000/evaluation/all/result_r05.pkl"

for p in (p1, p2):
    with open(p, "rb") as f:
        res = pickle.load(f)

    eer = res["eer"]
    arg = np.argmin(np.abs(res["precisions"] - eer))
    print(res["thresholds"][arg], " ", res["precisions"][arg], " ", res["recalls"][arg])
