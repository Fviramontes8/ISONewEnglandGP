from os import getcwd, makedirs, listdir
from os.path import exists, join


def validate_session_folder() -> None:
    """
    Creates a folder called 'sessions' where the main program runs

    Returns
    -------
    None
    """
    if (not exists(join(getcwd(), "sessions"))):
        makedirs(join(getcwd(), "sessions"), exist_ok=True)


def create_run_folder() -> str:
    """
    Creates folders necessary to store figures and logs for the training
     session

    Returns
    -------
    max_run_str: str
    String that contains the folder name to store figures and logs
    """
    validate_session_folder()
    session_folders: list[str] = listdir(join(getcwd(), "sessions"))
    max_run: int = -1
    for folder in session_folders:
        if folder.startswith("run"):
            current_run = int(folder[3:])
            if current_run > max_run:
                max_run = current_run
    max_run += 1
    makedirs(join(getcwd(), f"sessions/run{max_run}/figs"), exist_ok=True)
    max_run_str: str = f"run{max_run}"
    return max_run_str
