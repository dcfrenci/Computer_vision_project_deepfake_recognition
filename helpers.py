import os


def print_section(title, char='-'):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    print(f"{char * 3} {title} {char * (width - len(title) - 6)}")


def print_title(title, char='-'):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    half_width = int((width - len(title) - 2) / 2)
    print(f"{char * half_width} {title} {char * (width - half_width - len(title) - 3)}")
