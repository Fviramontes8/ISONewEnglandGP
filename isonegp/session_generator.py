from os import getcwd, makedirs, listdir
from os.path import exists, join


def validate_run_folder() -> None:
    """
    Creates a folder called 'runs' where the main program runs

    Returns
    -------
    None
    """
    if not exists(join(getcwd(), "runs")):
        makedirs(join(getcwd(), "runs"), exist_ok=True)


def create_run_folder() -> str:
    """
    Creates folders necessary to store figures and logs for the training
     session

    Returns
    -------
    max_run_str: str
    String that contains the folder name to store figures and logs
    """
    validate_run_folder()
    session_folders: list[str] = listdir(join(getcwd(), "runs"))
    max_run: int = -1
    for folder in session_folders:
        if folder.startswith("train"):
            current_run = int(folder[5:])
            if current_run > max_run:
                max_run = current_run
    max_run += 1
    makedirs(join(getcwd(), f"runs/train{max_run}/figs"), exist_ok=True)
    max_run_str: str = f"train{max_run}"
    return max_run_str
