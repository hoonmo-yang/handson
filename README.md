# Hands On

## 1. Python 기본 환경 세팅

### 개요
 AI-LLM을 돌리기 위한 기본적인 Python 환경을 세팅한다.   
 Python은 라이브러리의 개별적인 관리의 편리성 때문에 가상 환경화를 많이 사용한다.  
 그중 하나인 Miniforge 3 + Minimamba 조합을 설치해 본다.

1. 먼저 Miniforge 3 환경을 설치하기 위한 스크립트를 다운 받는다.  
    [주의 사항] $는 프롬프트를 나타내므로 실제 입력하는 명령어가 아니다.
```shell
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

 2. 다운로드 받은 스크립트를 실행한다.
 ```shell
 $ bash ./Miniforge3-Linux-x86_64.sh
 ```
 수행을 하면 설치는 대화형으로 진행된다. 적당한 문구를 타이핑하고 엔터를 치면 다음 단계로 넘어간다.  
 여기까지 설치했으면 `conda` 환경이 설치된 것이다.

 3. Conda 환경을 활성화시킨다. 
 ```shell
 $ source ~/.bashrc
 ```
쉘 프롬프트가 `(base)`로 표현될 것이다. 여기서 `base`는 conda에서 기본으로 생성되는 환경이다.  
이제 내가 사용하고자 하는 환경을 만들자. 만든 환경에서 자기가 마음껏 python 라이브러리를 설치할 수 있다.
필요에 따라 환경을 여러 개 만들 수도 있다.

4. 새로운 환경을 만든다. 만들 환경의 이름은 `handson`이고 설치될 기본 파이썬 버전은 3.12.4이다.
```shell
$ conda create --name handson python=3.12.5
```

4. `handson` 환경을 활성화시킨다.
```shell
$ conda activate handson
```
이후 `pip`로 설치되는 모든 파이썬 라이브러리는 `handson` 환경에 별도로 설치될 것이다.

7. `pip`로 파이썬 패키지인 `numpy`와 `pandas`를 설치해 본다.
``` shell
$ mamba install numpy
$ pip install pandas
```

8. `pytorch`를 설치한다. 파이토치는 텐서플로우와 함께 가장 많이 사용되는 딥러닝 프레임워크이다. 
GPU 버전으로 설치할 것이다.
``` shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

9. 파이토치에서 GPU가 잘 활성화 되었는지 확인해 본다. 파이썬을 쉘에서 수행하면 대화형 모드로 실행하게 된다.
``` shell
$ python
Python 3.12.5 | packaged by conda-forge | (main, Apr 15 2024, 18:38:13) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

10. `>>>` 프롬프트는 파이썬 콘솔 환경임을 나타낸다. 다음과 같이 파이썬 함수를 입력한다.

```shell
>>> import torch
>>> torch.cuda.is_available()
True
>>> quit()
```

모두 정상적으로 수행되고 `True`가 출력되면 GPU가 제대로 동작하고 있는 것이다. `quit()`을 수행해서
파이썬 수행을 끝낸다. 


## 2. LLM 기본 환경 세팅

### 개요
  오픈소스 LLM 모델 관리에 가장 많이 사용되는 `ollama`를 설치하고 동작해 본다.
  `ollama`의 사용법은 docker와 매우 유사하다. 공용 리포지스토리에서 원하는 모델을
  가져와서 (pulling) 직접 수행해 볼 수 있다.

1. `ollama`를 설치하기 위한 스크립트를 다운로드 받는다.
``` shell
$ wget https://ollama.com/install.sh 
```

2. 다운로드 받은 스크립트를 수행한다.
(주의 사항: 리눅스 머신에서는 어드민 계정 권한이 있어야 한다. 어드민 계정이 없으면 생략해도 된다.)
``` shell
$ bash ./install.sh
```

3. ollama 데몬이 수행 중인지 확인한다.
``` shell
$ systemctl status ollama
```

아래와 같이 `Active: active (running)`이란 문구가 뜨면 정상적으로 동작하는 것이다.

``` shell
$ systemctl status ollama
ollama.service - Ollama Service
     Loaded: loaded (/etc/systemd/system/ollama.service; enabled; preset: enabled)
     Active: active (running) since Thu 2024-08-29 11:34:31 KST; 4 days ago
   Main PID: 2255 (ollama)
      Tasks: 22 (limit: 38086)
     Memory: 6.1G (peak: 7.1G)
        CPU: 42.422s
     CGroup: /system.slice/ollama.service
             └─2255 /usr/local/bin/ollama serve
             ...
```

4. llama3.1 8B llm 모델을 풀링한다
``` shell
$ ollama pull llama3.1:latest
```

`ollama`를 이용하여 llama3.1 8B 모델을 다운로드하고 있다.

5. 다운로드 되었는지 `ollama list` 명령어로 확인해 본다.
``` shell
$ ollama list
NAME           	ID          	SIZE  	MODIFIED           
llama3.1:latest	f66fc8dc39ea	4.7 GB	About a minute ago	
```

llama3.1 모델이 다운로드 되었는지 확인할 수 있다.

6. `ollama run`으로 LLM을 수행해 본다
``` shell
$ ollama run llama3.1:latest
>>>
```

정상적으로 수행되면 `>>>` 프롬프트가 뜬다. 콘솔로 LLM과 대화를 나눌 수 있다.
대화를 나눠 본다.

``` shell
>>> 안녕?
하이! 어떻게 지냈어?
>>>
```

7. LLM을 종료하려면 `/bye`를 입력한다
``` shell
>>> /bye
```

8. 위와 같은 절차로 다른 LLM도 수행해 본다. (genmma2:latest, qwen2:latest)


## 3. Langchain 설치

### 개요
Langchain은 LLM의 응용을 위해 가장 많이 사용되는 프레임워크이다. 
LLM을 잘 적용하기 위해서는 langchain을 잘 아는 것이 매우 중요하다. 
langchain 설치를 수행한다.

1. langchain은 python이나 java-script 라이브러리 두 가지가 존재한다. 이중 
파이썬 라이브러리를 설치해 본다. 필요한 패키지를 다음과 같이 수행한다.

```shell
$ pip install langchain
$ pip install langchain_community
```

`langchain` 패키지는 기본 패키지이다. `langchain_community` 패키지는 
사용자 그룹이 만든 확장 패키지이다. 둘다 모두 필요하니 반드시 설치해야 한다.

2. Huggingface 패키지를 설치한다.
```shell
$ pip install huggingface_hub transformers datasets
```

Huggingface 패키지는 langchain 패키지는 아니나 LLM 응용을 위해서는 반드시
필요한 패키지라서 모두 설치해야 한다.

3. 몇가지 부가적인 패키지를 더 설치한다.
``` shell
$ pip install pymupdf
$ pip install sentence-transformers
```
`pymupdf`는 RAG 구성하기 위해 PDF 로딩을 위해 필요하다.
`sentence-transformers`는 로딩한 문서를 자르기 위해 (chunking) 필요하다.

4. RAG 구성을 위한 벡터 DB를 설치한다. 
``` shell
$ pip install chromadb
$ pip install faiss-gpu-cu12
```

5. langserve를 사용하기 위한 필요 패키지를 설치한다.
``` shell
$ pip install langserve
$ pip install fastapi uvicorn
```

## 4. 콘솔 기반 LLM 코딩 및 테스트
### 개요
* 문장 하나를 LLM에 입력하여 화면에 텍스트로 출력하는 간단한 예제를 작성한다
* 체인은 prompt, LLM, 문자열 출력 파서의 구조를 가진다.
* 출력이 한국어로만 나오도록 한다.
* 입력할 문장은 "한국의 수도는 어디니?"와 같은 형태로 파이썬 코드 상 문자열로
입력된다.

1. `basic.py`를 작성한다.
2. `basic.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.


## 5. Langchain 디버깅 환경 세팅 및 테스트
### 개요
* langsmith는 langchain 수행 시 로그를 출력하는 서비스이다.
* langsmith를 사용하려면 먼저 API 키를 발급 받아야 한다.
* langsmith를 세팅하여 앞서 콘솔 기반 LLM을 디버깅해 본다.

1. Langsmith API 키 발급
2. Langsmith를 코드에 적용하여 수행
3. Langsmith UI 화면에서 디버깅

## 6. Langserve UI 기반 LLM 코딩 및 테스트
### 개요
* Langserver는 개발자가 간단하게 테스트해 볼 수 있는 Web UI이다.
* langserve UI를 사용하여 LLM 예제를 작성한다.
* FastAPI, uvicorn, langserve의 add_routes 함수를 이용하여 구축한다.

1. `langserve_ui.py`를 작성한다.
2. `langserve_ui.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.

## 7. 번역기 구현 (Prompt 튜닝)
### 개요
* 프롬프트를 제어하여 LLM을 목적에 맞게 변경할 수 있다.
* 프롬프트를 작성해서 입력한 문장에 대해 답변을 하는 것이 아니라 영어로 번역을 하는
  간단한 번역기를 만들어 본다.
* langserve UI playground에서 동작 확인할 수 있게 작성한다.

1. `translate.py`를 작성한다.
2. `translate.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.

## 8. 문서요약기 구현 (Prompt 튜닝)
### 개요
* 프롬프트를 제어하여 LLM을 목적에 맞게 변경할 수 있다.
* 프롬프트를 작성해서 입력한 문장에 대해 답변을 하는 것이 아니라 입력한 
  문장을 요약하는 간단한 요약기를 만들어 본다.
* 모든 출력은 한국어로 되게 한다.
* langserve UI playground에서 동작 확인할 수 있게 작성한다.

1. `summary.py`를 작성한다.
2. `summary.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.

## 9. Langserve UI 기반 챗봇 코딩 및 테스트
### 개요
* 챗봇을 만들기 위해서는 대화를 별도로 저장하는 히스토리 기능을 구현해야 한다.
* 그러나 아주 간단한 응용일 경우, langserve UI만으로도 챗봇 구현이 가능하다.
  langserve가 알아서 히스토리 처리를 해준다.
* langserve 기반 챗봇을 작성한다.
* 모든 출력은 한국어로 되게 한다.
* langserve playground에서 챗봇을 작동해 본다.

1. `chat.py`를 작성한다.
2. `chat.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.

## 10. RAG 구축 LLM 코딩 및 테스트
### 개요
* RAG를 구현해 본다.
* PDF 문서를 찾는다. (법률이나 어떠한 문서라도 좋다)
* PDF 문서를 로딩, 임베딩, 청킹 (자르기)를 하여 벡터 DB에 저장한다.
* 이후 Retriever를 연동하여 RAG + LLM을 구축한다.
* RAG가 있는 경우와 없는 경우 LLM의 동작이 어떻게 다른지 확인한다.

1. pdf로 RAG에 로딩할 문서를 생성한다.
2. `rag.py`를 작성한다.
3. `rag.py`를 수행한다.
4. `langsmith`를 이용하여 디버깅을 해 본다.

## 11. 대화형 RAG를 구축한다. (RAG + LLM Chatbot)
### 개요
앞서 작성한 챗봇은 RAG가 연동된 구현이 아니다.
또한 앞서 구현한 RAG 예제는 히스토리 기능이 없어서 대화를 할 수 없다.
이 둘을 합친 대화형 RAG를 구축한다.

* RAG를 구축한다.
* 챗봇을 구축한다.
* RAG와 챗봇을 연동한 대화형 RAG를 구축한다.
* RAG가 있는 경우와 없는 경우 LLM의 동작이 어떻게 다른지 확인한다.

1. `icrag.py`를 작성한다.
2. `icrag.py`를 수행한다.
3. `langsmith`를 이용하여 디버깅을 해 본다.
