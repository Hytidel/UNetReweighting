from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol, Any


def cal_psnr(
    ref_img_path: Union[str, List[str]], 
    gen_img_path: Union[str, List[str]]
) -> Union[
    float,  # single: psnr_score
    Tuple[List[float], float]  # multi: (psnr_score_list, avg_psnr_score)
]:
    single = False

    if isinstance(ref_img_path, str):
        ref_img_path_list = [ref_img_path]
        single = True
    else:
        ref_img_path_list = ref_img_path
    if isinstance(gen_img_path, str):
        gen_img_path_list = [gen_img_path]
    else:
        gen_img_path_list = gen_img_path
    
    if len(ref_img_path_list) != len(gen_img_path_list):
        raise ValueError(
            f"The length of `ref_img_path_list` doesn't match the length of `gen_img_path_list`, "
            f"got {len(ref_img_path_list)} and {len(gen_img_path_list)}. "
        )

    num_img_pair = len(ref_img_path_list)

    import cv2
    from skimage.metrics import peak_signal_noise_ratio

    psnr_score_list = []
    for (ref_img_path, gen_img_path) in zip(
        ref_img_path_list, 
        gen_img_path_list
    ):
        ref_img_cv2 = cv2.imread(ref_img_path)
        gen_img_cv2 = cv2.imread(gen_img_path)
        
        tmp_psnr_score = peak_signal_noise_ratio(ref_img_cv2, gen_img_cv2)
        psnr_score_list.append(tmp_psnr_score)

    if single:
        return psnr_score_list[0]

    avg_psnr_score = sum(psnr_score_list) / num_img_pair

    return psnr_score_list, avg_psnr_score
    