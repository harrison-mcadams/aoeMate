import getSS

# AOEMate -- Python code that analyzes live-streaming AOE4 game. First goal is to monitor the production queue to
# warn if no villages are being made.

if __name__ == "__main__":

    # Set where to look on the screen
    gfn_region = {'top': 600, 'left': 000, 'width': 800, 'height': 600}

    # get screenshot
    screenshot = getSS.capture_gfn_screen_region(gfn_region)

    # analyze screenshot

