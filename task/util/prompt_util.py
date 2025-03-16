from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import re


def preprocess_prompt(
    prompt: str
) -> str:
    return re.sub(
        r"[^a-zA-Z0-9,. ]", '', 
        prompt
    )

def get_folder_name(
    prompt: str, 
    used_folder_name_list: List[str]
) -> str:
    prompt = re.sub(
        r"[^a-zA-Z0-9 ]", '', 
        prompt
    )
    word_list = prompt.split()

    num_word = 4
    while num_word < len(word_list):
        folder_name = word_list[: num_word]
        folder_name = '_'.join(
            [word.lower() for word in folder_name]
        )

        if folder_name not in used_folder_name_list:
            used_folder_name_list.append(folder_name)

            return folder_name
        else:
            num_word += 1

    folder_name = '_'.join(
        [word.lower() for word in word_list]
    )
    used_folder_name_list.append(folder_name)

    return folder_name
