# terminal-minecraft
Simple minecraft clone but it runs in the windows terminal.
Works by using the foreground+background trick with the half block character ("▄"), simulating two "pixels" per terminal character.
As much as i tried to optimize it, it requires a decent cpu to run "normally", if you have a low-end pc, please check the settings and update the optimization values accordingly.
Since the render is what causes most spikes, consider changing the render distance and increase the terminal font size for quick testing, though I recommend checking out the settings.
Additionally, if you have above-medium / high end harware, also change the settings to the high ones for faster calculations, generation and rendering etcetc

# showcase
<img src="https://github.com/user-attachments/assets/e556e16c-5869-4c81-93fd-13a3e883d13c" width="800">
<img src="https://github.com/user-attachments/assets/e1a3d29f-60dd-47ff-8444-a4691e17174f" width="400">
<img src="https://github.com/user-attachments/assets/a728239c-0904-4313-9d9c-bc2aa7314bf1" width="400">

*textured gameplay  /  my lagoon hut, very pretty i know*

NOTE: Beware that quality scales up with the window size and font size, but so does the impact on the cpu!


# controls
* mouse : camera look
* WASD : movement
* Ctrl^C : close  ( cleanup may take a few seconds )
* TAB : toggle flight mode
* C : move down

Notice: Beware that the mouse will be thread locked when chunk pregeneration finishes and will only be released when the program is closed


# diagram
![Untitled Diagram drawio(1)](https://github.com/user-attachments/assets/6e2dacbf-65f1-4746-9931-352bf4dda993)
