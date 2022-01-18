# vscode-hpc
This is a sample code accompanying SHARCNET General Interest Seminar entitled: *Remote Development on HPC Clusters with VSCode*. You can find a recorded version of the talk on the [SHARCNET YouTube channel](https://www.youtube.com/watch?v=u9k6HikDyqk).

## The setup

* [Visual Studio Code (VSCode)](https://code.visualstudio.com/) on the system that you work on it along with the following extensions:
    - [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
    - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
        * [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
        * [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) (*optional dependency*)
        * [Visual Studio IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode) (*optional*)
    - [Nsight Visual Studio Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)
    - [Makefile Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.makefile-tools) (*optional*)
    - [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github) (*optional*)
    - [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens) (*optional*)

And the followings on the platform(s) that you want to do either *local* or *remote* development:
### Compute Canada Clusters
For remote development on *Compute Canada* clusters, the following ```module``` command will do the trick but often you have to add it at the end of your ```~/.bashrc``` file:

```
module load cmake cuda scipy-stack/2020a ipykernel
```
### Linux
* C++ compiler supporting the ```C++14``` standard (e.g. ```gcc``` 9.3)
* [Python 3](https://www.python.org/downloads/)
* [Git](https://git-scm.com/download/linux) for *Linux*
* [CMake](https://cmake.org/) 3.18 or higher for *Linux*
* An MPI implementation (e.g. ```OpenMPI``` or ```MPICH```)
* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux) for *Linux*

### Windows
* [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/) with ```C++``` and ```Python``` support
* [Git](https://git-scm.com/download/win) for *Windows*
* [CMake](https://cmake.org/download/) 3.18 or higher for *Windows*
* [Windows Terminal](https://aka.ms/terminal) or [Windows Terminal Preview](https://aka.ms/terminal-preview)
* [MS-MPI](https://www.microsoft.com/en-us/download/details.aspx?id=100593) (both ```msmpisetup.exe``` and ```	
msmpisdk.msi```)
* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) for *Windows*

On Windows systems, if you want to do both local development on *Windows* and remote development on [WSL2](https://docs.microsoft.com/en-us/windows/wsl/), you have to first install the [NVIDIA drivers for WSL with CUDA and DirectML support](https://developer.nvidia.com/cuda/wsl/download) on *Windows* and then follow these [instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#setting-up-linux-dev-env) in order to install ```CUDA toolkit``` on *WSL2*.

### macOS
* C++ compiler supporting the ```C++14``` standard (e.g. ```clang``` 3.4)
* [Python 3](https://www.python.org/downloads/)
* [Git](https://git-scm.com/download/mac) for *macOS*
* [CMake](https://cmake.org/download/) 3.18 or higher for *macOS*
* An MPI implementation (e.g. ```OpenMPI``` or ```MPICH```)
* [CUDA toolkit](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts) for *macOS*

## Get started
Just run *VSCode* on the system that you work on it and then select ```Clone Git Repository...``` from ```Get Started``` page or type ```git: clone``` in the *command palette* (<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>p</kbd>  or <kbd>F1</kbd> key). Then paste ```https://github.com/sharcnet/vscode-hpc.git``` and hit ```Enter```.
