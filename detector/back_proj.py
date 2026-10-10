'''
Script to train and test detector model on frames1
'''

def backproject(depth, intrinsics):
    #depth are of H x W
    #intrinsics are of 3x3 fx,fy,cx,cy
    #essentially compte pixels x,y coordinates within a given frame
    #(vectorizing for your case) and then compute that pixels depth in the frame
    #from depth maps and intrinsics
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)  # H x W x 3
    return points

