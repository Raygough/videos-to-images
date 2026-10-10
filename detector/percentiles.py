import json, random

with open('/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_to_sequence_map_v3.json') as f:
    mapping = json.load(f)

THRESHOLD = 150

kept = {k: v for k, v in mapping.items() if v.get('mse') is not None and v['mse'] <= THRESHOLD}
dropped = {k: v for k, v in mapping.items() if k not in kept}

print('Kept: {} frames'.format(len(kept)))
print('Dropped: {} frames'.format(len(dropped)))

with open('/nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_to_sequence_map_final.json', 'w') as f:
    json.dump(kept, f, indent=2)


near_threshold = [k for k, v in mapping.items() if v.get('mse') and 130 <= v['mse'] <= 170]
print(random.sample(near_threshold, min(4, len(near_threshold))))