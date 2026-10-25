# rumz // 6/7/24

### Feb 2025
### NOT IN USE

from spectral import *

def variance(corrected_img, pc_num):
    print("Shape: ", corrected_img.shape)
    pc = principal_components(corrected_img)
    pc_reduce = pc.reduce(num= pc_num)
    x = len(pc_reduce.eigenvalues)
    print("PCA num: ", x)
    img_pc = pc_reduce.transform(corrected_img)
    #imshow(img_pc, (0, 1, 2), title="pc image 1")
    imshow(img_pc,(0,1,2),stretch_all=True, title="pc image 1",  figsize=[24,36])
    if x > 3:
        step2 = imshow(img_pc, (1, 2, 3), stretch_all=True, title= 'pc image 2', figsize=[24,36])
        #save_rgb('CMC1-scan6_PCA.tif', step2, [1, 2, 3])
    #view = imshow(img_pc, (2, 3, x))
    #print("pc image shape: ", img_pc.shape)

    return img_pc