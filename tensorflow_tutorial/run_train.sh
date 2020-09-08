#Set current directory
CURR_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYENV_NAME="$(basename "$CURR_DIR")"

echo "Building python environment"

#parse arguments
GPU=false
while true; do
  case "$1" in
    -g | --use_gpu ) GPU=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

pyenv virtualenv --force 3.7.1 "pyenv.$PYENV_NAME"
pyenv local pyenv.$PYENV_NAME
pip install --upgrade pip setuptools

if [ $GPU == true ] ;
then
  echo "Using GPU version of Tensorflow"
  pip install -r requirements/gpu.txt
else
  pip install -r requirements/no_gpu.txt
fi

echo "Running training"
python train_model.py