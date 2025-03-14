# AIServer

A local AI API server compatible with OpenAI, written in Python.

- Uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python), [pywhispercpp](https://github.com/absadiki/pywhispercpp), [piper-tts](https://github.com/rhasspy/piper) as backend for AI services.

# Installation

### 1. Requirements:

  - Python 3.11+ (3.11 is recommanded and tested)
  - C compiler
      - Linux: gcc or clang
      - Windows: Visual Studio or MinGW
      - MacOS: Xcode
  - ffmpeg library (Ubuntu Linux: `sudo apt install ffmpeg`)
  - git

### 2. Clone this repository.

To clone this repository, run:

```bash
git clone https://github.com/smbhuin/aiserver.git
cd aiserver
```

### 3. Install python dependencies

First create virtual environment with python 3.11 (Recommanded) and activate.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To install the basic python dependencies, run:

```bash
pip install -r requirements.txt
```

To install individual backend packages, follow their own instructions.

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation)
- [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python?tab=readme-ov-file#installation)
- [pywhispercpp](https://github.com/absadiki/pywhispercpp?tab=readme-ov-file#installation)
- [piper-tt](https://github.com/rhasspy/piper?tab=readme-ov-file#installation)

or to install pre-build cpu packages:

```bash
pip install -r requirements_cpu.txt
```

or to install gpu cuda packages:

```bash
pip install -r requirements_gpu.txt
```

To use transcribe api with files other than wav, you need to install ffmpeg: 

```bash
sudo apt install ffmpeg
```

### 4. Download models

Download your prefered models supported by backends used in this project.

For example:

- LLM Models
  - [Mistral 7B Instruct GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
  - [Llama 3.1 8B Instruct GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

- SD/SDXL/Flux Models
  - [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

- Whisper Models
  - [Whisper GGML](https://huggingface.co/ggerganov/whisper.cpp)

- Piper TTS Voices
  - [Piper Voices](https://github.com/rhasspy/piper/blob/master/VOICES.md)

After downloading all the models `models` dir shall look like folowing:

```
models
│   qwen2.5-coder-7b-instruct-q5_k_m.gguf
│   mistral-7b-instruct-v0.2.Q4_K_M.gguf
|   nomic-embed-text-v1.5.f16.gguf
│
└───stable-diffusion
│   │   v1-5-pruned-emaonly.safetensors
│   │   dreamshaper_8.safetensors
│
└───audio
    │   en_US-amy-medium.onnx
    │   en_US-amy-medium.onnx.json
    |   whisper-small-ggml.bin
    |
    |
```

### 5. Create the config file

Rename aiserver.example.yml to aiserver.yml and edit to include your models.

### 6. Run the API server

`python main.py`

# Usage

Open `http://localhost:8000/docs` for OpenAPI docs.

**Chat Completion:**

```python
from openai import OpenAI
client = OpenAI()
openai.api_base = 'http://localhost:8000'

completion = client.chat.completions.create(
  model="mistral-7b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

**Image Generation:**

```python
from openai import OpenAI
client = OpenAI()
openai.api_base = 'http://localhost:8000'

response = client.images.generate(
    model="dreamshaper-8",
    prompt="a white siamese cat",
    size="512x512",
    quality="standard",
    n=1,
)

print(response.data[0].url)
```

**Speech To Text:**

```python
from openai import OpenAI
client = OpenAI()
openai.api_base = 'http://localhost:8000'

audio_file= open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
)

print(transcription.text)
```

**Text To Speech:**

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is a wonderful day to build something people love!",
)
response.stream_to_file(speech_file_path)
```

# Discussions and contributions

If you find any bug, please open an [issue](https://github.com/smbhuin/aiserver/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/smbhuin/aiserver/discussions) and open a new topic.

# Credits

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [openedai-images](https://github.com/matatonic/openedai-images)

# License

This project is licensed under [MIT License](./LICENSE.md).
