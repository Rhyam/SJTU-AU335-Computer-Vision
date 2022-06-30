import cv2
import numpy as np
from glob import glob

save_path = 'tolabel/'
i = 329
with open('name_label.txt', 'a', encoding= 'utf-8') as file:
    for name in glob('green/*'):
        img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512))
        save_name = save_path + '{}.jpg'.format(i)
        cv2.imwrite(save_name, img)
        #file.write(name[7:] + '_{}.jpg\n'.format(i))
        i += 1