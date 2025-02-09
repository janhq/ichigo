# Install and Upload Package into PyPI

## Requirements

* Python=3.10
* Dependencies: build, twine

## Local

```bash
python -m build
pip install dist/ichigo-0.0.1-py3-none-any.whl
python -c "import ichigo.asr as asr; print(asr.__file__)" 
python -c "from ichigo.asr import transcribe; results = transcribe('sample.wav'); print(results)"
python -c "from ichigo.asr import get_stoks; stoks = get_stoks('speech.wav'); print(stoks)"
```

## Upload into PyPI

```
python -m build
python -m twine upload dist/* 
```