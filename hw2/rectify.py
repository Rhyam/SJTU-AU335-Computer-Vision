import numpy as np
import cv2

def get_ARt(P):

    Q = np.linalg.inv(P[0:3, 0:3])
    U, B = np.linalg.qr(Q)
    R = np.linalg.inv(U)
    t = B @ P[0:3, 3]
    A = np.linalg.inv(B)
    A = A / A[2,2]

    return A, R, t

def rectify(Po1, Po2):

    A1, R1, t1 = get_ARt(Po1)
    A2, R2, t2 = get_ARt(Po2)

    c1 = -np.linalg.inv(Po1[:, 0:3]) @ Po1[:, 3]
    c2 = -np.linalg.inv(Po2[:, 0:3]) @ Po2[:, 3]

    # new world axis
    v1 = (c1 - c2)
    v2 = np.cross(R1[2, :].T, v1)
    v3 = np.cross(v1, v2)

    # rotation matrix
    R = np.array([v1.T / np.linalg.norm(v1),
                  v2.T / np.linalg.norm(v2),
                  v3.T / np.linalg.norm(v3)])

    # intrinsic matrix
    A = (A1 + A2) / 2
    A[0, 1] = 0 # avoid skew

    # projection matrix
    Pn1 = A @ np.append(R, np.array([-R @ c1]).T, axis=1)
    Pn2 = A @ np.append(R, np.array([-R @ c2]).T, axis=1)

    # rectify image transformation
    T1 = Pn1[0:3, 0:3] @ np.linalg.inv(Po1[0:3, 0:3])
    T2 = Pn2[0:3, 0:3] @ np.linalg.inv(Po2[0:3, 0:3])

    return T1, T2, Pn1, Pn2

if __name__ == "__main__":
    
    img_shape = (768, 576)
    Po1 = np.array([[976.5, 53.82, -239.8, 387500],
                    [98.49, 933.3, 157.4, 242800],
                    [0.579, 0.1108, 0.8077, 1118]])
    Po2 = np.array([[976.7, 53.76, -240.0, 40030],
                    [98.68, 931.01, 156.71, 251700],
                    [0.5766, 0.11411, 0.8089, 1174]])
    T1, T2, Pn1, Pn2 = rectify(Po1, Po2)
    
    img1 = cv2.imread('left.png')
    img2 = cv2.imread('right.png')
    img1_warped = cv2.warpPerspective(img1, T1, img_shape, cv2.INTER_LANCZOS4)
    img2_warped = cv2.warpPerspective(img2, T2, img_shape, cv2.INTER_LANCZOS4)

    cv2.imshow('img1_origin', img1)
    cv2.imshow('img2_origin', img2)
    cv2.imshow('img1_rectified', img1_warped)
    cv2.imshow('img2_rectified', img2_warped)

    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
