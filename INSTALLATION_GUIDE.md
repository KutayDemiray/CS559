# Steps to install

- First, follow SNeRL's instructions (README.md). Be careful not to install a newer version of Metaworld, it won't work.
- You'll (probably) get ```cymj.pyx``` compile error: Do ```pip install cython==0.29.36``` and try again, apparently newer versions don't work.

## Common issues

### Failed to load XML due to missing texture

You get an error like ```Error: PNG file load error```.

For some reason this repo did not originally have all texture files, fetch them from modern Metaworld and add to the appropriate place.

### Failed to initialize OpenGL

I'm not sure what causes this error exactly down the line.

#### From ```env.render()``` (human mode)

One suggestion I found on the internet was adding

```
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
```

to the beginning of the Python file. However, this did not really solve it when I first tried it, but after some time the problem magically solved itself.

It might be related to remote desktops (I wasn't using it when I got the error, but maybe if someone else is connected to another user at the same time)

#### From ```env.sim.render()``` (trying to get RGB image)
