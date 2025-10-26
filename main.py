import getSS
import analyzeSS
from PIL import Image

# AOEMate -- Python code that analyzes live-streaming AOE4 game. First goal is to monitor the production queue to
# warn if no villages are being made.

if __name__ == "__main__":

    for ii in range(1000000):

        # Set where to look on the screen
        gfn_region = {'top': 850, 'left': 0, 'width': 300, 'height': 350}

        # get screenshot
        screenshot = getSS.capture_gfn_screen_region(gfn_region)



        vill_kernel = Image.open("/Users/harrisonmcadams/Desktop/villager_icon.png")

        out_path = '/Users/harrisonmcadams/Desktop/'

        convolved_image = analyzeSS.convolveSSbyKernel(screenshot, vill_kernel)

        binary = analyzeSS.isTargetInSS(convolved_image, vill_kernel)

        if binary:
            print('Villagers are producing!')
        else:
            print('Villagers are NOT producing! :-(')

