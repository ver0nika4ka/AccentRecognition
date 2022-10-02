# Accent Recognition

Accent recognition in foreign language speech using [Speech Accent Archive Dataset](https://accent.gmu.edu/howto.php). A 2-layer CNN is used as a model.
A variety of audio signal features can be extracted with an option to remove silence fragments from inspected audio samples.
The code also allows flexible parameters adjustments.


## Setup
The steps below were tested with Python 3.8 (Ubuntu) and Python 3.9 and 3.10 (macOS).

### Clone Repository and Install Python Dependencies
1. Open the Terminal (or Command Prompt on Windows), clone the repository and navigate into the directory:
`git clone https://github.com/ver0nika4ka/AccentRecognition.git && cd AccentRecognition` 
2. Create a virtual environment and activate it:
- ***Unix-based system***:
    `python3 -m venv accent-env && source accent-env/bin/activate`
- ***Windows*** (note: on Windows type `python` instead of `python3` here and for the steps below):
    `python -m venv accent-env && accent-env/Scripts/activate.bat`

3. Update `pip` and install requirements except TensorFlow:
    `python3 -m pip install -U pip setuptools wheel && python -m pip install -r requirements.txt`

4. Install TensorFlow version according to the operating system and its architecture:
- ***macOS with Apple Silicon (M1, M2, etc.)***:
    `python3 -m pip install tensorflow-macos==2.10 tensorflow-metal==0.6`
- ***All other Operating Systems and architectures (Ubuntu, Windows, macOS with Intel processor, etc.)***:
    `python3 -m pip install tensorflow==2.10`

### Set up Comet ML
[Comet ML](https://www.comet.com/site/) is needed for visualization and logging of the experiments. 
Follow [the official guide](https://www.comet.com/docs/v2/guides/getting-started/quickstart/) 
on how to create an account (we recommend to Sign Up with GitHub) and get *API_KEY*. Next, create a new project (e.g., accent-recognition).
If you used GitHub to sign up, the workspace name is your GitHub username. Next, set environment variables in the Terminal:

***Unix-based systems***. Open Terminal and run: 
```shell
export COMET_API_KEY="Your API Key"
export COMET_WORKSPACE="Your Workspace Name"
export COMET_PROJECT_NAME="Your Project Name"
```

***Windows***. Open Command Prompt and run:
```commandline
set COMET_API_KEY "Your API Key"
set COMET_WORKSPACE "Your Workspace Name"
set COMET_PROJECT_NAME "Your Project Name"
```

If you're using IDE for development, consult relevant documentation on how to set up 
"Run configurations" for the project with specific environment variables (e.g., 
[PyCharm](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html), 
[Visual Studio Code](https://code.visualstudio.com/docs/python/environments#_environment-variables)).

Alternatively, you can set these variables manually in the code (see `build_model.py` for details).

## Usage
Before proceeding with the steps below, make sure that you activated the virtual environment. If you've just set it up according to the instructions above you're good to go. If you return to the project after restarting the Terminal or Command Prompt, navigate to the project directory and run `source accent-env/bin/activate` on Unix or `accent-env/Scripts/activate.bat` on Windows to activate the environment.

### (Optional) Copying downloaded and preprocessed data
If you have an archive with the downloaded and converted data, to speed things up, you can extract the archive into the root of the directory. It will create `audios` directory and `metadata.json` file.

### Get and Preprocess Data (Not required if you've already extracted the downloaded and preprocessed data)
To download the audio files from speech archive run:
```shell
python3 get_data.py
```

It will download all the languages that are defined in the dictionary `LANGUAGES` in `constants.py`. The current list of languages is as follows:
```python
languages = ['arabic', 'bengali', 'bulgarian', 'chinese', 'dari', 'dutch',
             'english', 'french', 'german', 'gujarati', 'hindi', 'italian',
             'kurdish', 'macedonian', 'nepali', 'pashto', 'polish', 'portuguese',
             'romanian', 'russian', 'spanish', 'swedish', 'tajiki', 'urdu']
```
All the available languages can be found [here](https://accent.gmu.edu/browse_language.php). When adding a new language, define a unique two-letter key (e.g., `en` for English) that will be used in `LANG_SET` variable in `build_model.py`. 

To preprocess data (i.e., to convert `.mp3` files to `.wav` and remove silence) run the following code. ***Important***: To perform data preprocessing you have to install `ffmpeg`. On Ubuntu, it can be done via `sudo apt update && sudo apt install ffmpeg`; on Mac: `brew install ffmpeg` (but you must have [brew](https://brew.sh/) installed). On Windows it requires more steps, one of the guides can be found [here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/).
```shell
python3 preprocess_data.py
```

### Build and Train Model
Once you have all the data downloaded and preprocessed, you can run feature extraction and training:
```shell
python3 build_model.py
```
See `Parameters to adjust` section in `build_model.py` on how to change the language sets and specify what features to extract and use for training.
