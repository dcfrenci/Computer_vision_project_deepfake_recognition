import os


def print_section(title, char='-'):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    print(f"{char * 3} {title} {char * (width - len(title) - 6)}")
