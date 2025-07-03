# terminal-minecraft
Simple minecraft clone but it runs in the windows terminal.
Works by using the foreground+background trick with the half block character ("▄"), simulating two "pixels" per terminal character.
As much as i tried to optimize it, it requires a decent cpu to run "normally", if you have a low-end pc, please check the settings and update the optimization values accordingly.
Since the render is what causes most spikes, consider changing the render distance and increase the terminal font size for quick testing, though I recommend checking out the settings.
Additionally, if you have above-medium / high end harware, also change the settings to the high ones for faster calculations, generation and rendering etcetc

# showcase
<img src="https://github.com/user-attachments/assets/558d6d46-7782-481e-aef4-ea3971560e06" width="400">
<img src="https://github.com/user-attachments/assets/a728239c-0904-4313-9d9c-bc2aa7314bf1" width="400">

*windowed gameplay  /  fullscreen (with my lagoon hut, very pretty i know)*

NOTE: Beware that quality scales up with the window size and font size, but so does the impact on the cpu!


# controls
* mouse : camera look
* WASD : movement
* Ctrl^C : close  ( cleanup may take a few seconds )
* TAB : toggle flight mode
* C : move down

Notice: Beware that the mouse will be thread locked when chunk pregeneration finishes and will only be released when the program is closed


# updates
v 1.0.0 : Basic Clone with curses rendering
* Showcase: https://www.youtube.com/watch?v=9djN_DJn6x0&t
* Procedural chunk generation ( 2 biomes : plains, forest )
* Cave and tree generation (mostly random)
* Windows curses render from opengl context
* Primitive mouse draw controller
* WASD controller and keylisteners
* Voxel culling and frustum
* total of 10 block types

v 1.2.0  ( CURRENT )
* Showcase: https://www.youtube.com/watch?v=7TLsNwHLdWw&t
* Smoother and lazyer chunk generation, load and unloading
* Mesh caching and faster mesh updates
* Block placement/breaking/selection
* Block hotbar
* Support for more color combinations
* Improved optimization settings for low-medium end pcs
* Decent mouse look controller
* Logo title thingy

# todo
* ✗ Add dirty chunk memory
* ✗ Add world save and load features
* ✓ Fix stupid ahh raytrace offsets
* ✓ Reduce memory load on mesh render
* ✓ Cache colors and rendered data, draw only changes
* ✓ Cache rencently visited world data
* ✓ Use lazy, position based generatiors with priority for less intense generation
* ✓ Fix color combination out of bounds artifacts
* ✗ Add an actuall title screen
* ✗ Add propper gui like pause menu, settings, etc
* ✗ Add an actuall chat and achievements
* ✗ Fix water broken face culling at chunk edges
* ✗ Paint x,-x and z,-z faces of different colors to distinguish them
* ✓ Block hotbar
* Improved on preformance and caching
