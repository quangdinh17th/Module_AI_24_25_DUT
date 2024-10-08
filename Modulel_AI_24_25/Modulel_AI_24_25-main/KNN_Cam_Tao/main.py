import numpy as np

cam = np.array([2.7887, 6.5063, 9.4425, 9.8402, -19.5930])
tao = np.array([2.6743, 5.7745, 9.9031, 11.0016, -21.4722])
# test = numpy.array([2.6588, 5.7358, 9.6682, 10.7427, -20.9914]) 

test = np.array(list(map(float, input("Type input vector test: ").split())))

d_cam_test_norm1 = np.sum(np.abs(cam - test))
d_tao_test_norm1 = np.sum(np.abs(tao - test))

d_cam_test_norm2 = np.sqrt(np.sum(np.square(np.abs(cam - test))))
d_tao_test_norm2 = np.sqrt(np.sum(np.square(np.abs(tao - test))))

print("\nManhattan distance - Norm 1")
print("\tCam_Test:", round(d_cam_test_norm1, 4))
print("\tTao_Test:", round(d_tao_test_norm1, 4))

print("Euclidean distance - Norm 2")
print("\tCam_Test:", round(d_cam_test_norm2, 4))
print("\tTao_Test:", round(d_tao_test_norm2, 4), end='\n\n')

if (d_cam_test_norm1 < d_tao_test_norm1):
    print("Buc anh co the la Cam theo Norm 1")
else:
    print("Buc anh co the la Tao theo Norm 1")

if (d_cam_test_norm2 < d_tao_test_norm2):
    print("Buc anh co the la Cam theo Norm 2")
else:
    print("Buc anh co the la Tao theo Norm 2")

