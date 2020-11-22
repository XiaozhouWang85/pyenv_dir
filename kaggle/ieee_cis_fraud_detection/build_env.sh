#Set current directory
CURR_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYENV_NAME="$(basename "$CURR_DIR")"

echo "Building python environment"

pyenv virtualenv --force 3.7.1 "pyenv.$PYENV_NAME"
pyenv local pyenv.$PYENV_NAME
pip install --upgrade pip setuptools
pip install -r requirements.txt

echo "Running training"
python simple_train.py