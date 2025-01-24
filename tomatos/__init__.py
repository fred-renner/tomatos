# ASCII Art
ascii_art = [
    r" ______  ______   __    __   ______   ______  ______   ______    ",
    r"/\__  _\/\  __ \ /\ '-./  \ /\  __ \ /\__  _\/\  __ \ /\  ___\   ",
    r"\/_/\ \/\ \ \/\ \\ \ \-./\ \\ \  __ \\/_/\ \/\ \ \/\ \\ \___  \  ",
    r"   \ \_\ \ \_____\\ \_\ \ \_\\ \_\ \_\  \ \_\ \ \_____\\/\_____\ ",
    r"    \/_/  \/_____/ \/_/  \/_/ \/_/\/_/   \/_/  \/_____/ \/_____/ ",
    # r"                                                                 ",
]

# ANSI escape codes for colors (fading from red to cyan)
fade_colors = [
    "\033[91m",  # Red
    "\033[93m",  # Yellow
    "\033[92m",  # Green
    "\033[94m",  # Blue
    "\033[95m",  # Magenta
    "\033[96m",  # Cyan
]

# ANSI escape codes for the solid yellow frame and reset color
yellow = "\033[93m"
reset = "\033[0m"

# Determine frame width based on the longest line
max_width = max(len(line) for line in ascii_art)
frame_width = max_width + 4  # Add padding


# Function to apply fading colors from top to bottom
def fade_text_by_line(ascii_art_lines):
    faded_lines = []
    for i, line in enumerate(ascii_art_lines):
        color = fade_colors[i % len(fade_colors)]  # Pick color based on line index
        faded_lines.append(color + line + reset)
    return faded_lines


# Apply fade effect to each line of the ASCII art
faded_ascii_art = fade_text_by_line(ascii_art)

# Constructing the frame with solid yellow
top_border = yellow + "╭" + "─" * frame_width + "╮" + reset  # Top border
bottom_border = yellow + "╰" + "─" * frame_width + "╯" + reset  # Bottom border
bottom_border_ = yellow + "─" * (frame_width + 2) + reset  # Bottom border

# Print framed and colored ASCII art with padding
print(top_border)
print(
    f"{yellow}│{reset}" + " " * (frame_width) + f"{yellow}│{reset}"
)  # Empty line for spacing
for line in faded_ascii_art:
    padded_line = f"  {line}  "  # Add padding spaces
    print(f"{yellow}│{reset}{padded_line.ljust(frame_width)}{yellow}│{reset}")
print(
    f"{yellow}│{reset}" + " " * (frame_width) + f"{yellow}│{reset}"
)  # Empty line for spacing
print(bottom_border)
