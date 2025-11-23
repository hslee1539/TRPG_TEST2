# LangChain TRPG Game Master

이 프로젝트는 LangChain을 사용하여 LLM이 게임 마스터 역할을 수행하는 TRPG(테이블탑 롤플레잉 게임) 세션을 실행하는 커맨드라인 및 웹 도구입니다.

## 주요 구성 요소
- `trpg.game_master.GameMaster`: LangChain `LLMChain`을 감싸 플레이어 입력과 스토리 진행을 관리합니다.
- `trpg.game_master.GameState`: 현재까지의 사실(facts)을 저장하고 LLM 프롬프트에 제공합니다.
- `trpg.game_master.create_default_game_master`: 기본 시스템 프롬프트와 메모리를 사용하여 따뜻하고 서사적인 진행을 담당하는 게임 마스터를 생성합니다.
- `main.py`: 커맨드라인 인터페이스로, 플레이어 입력을 받아 LLM 응답을 출력합니다.
- `server.py`: 표준 라이브러리 `http.server`를 활용한 경량 웹 서버로, 브라우저에서 TRPG 세션을 플레이할 수 있는 간단한 UI를 제공합니다.

## 실행 방법
1. 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
2. LM Studio에서 원하는 모델을 선택하고 "OpenAI Compatible Server"를 실행합니다.
   - 기본적으로 서버는 `http://localhost:1234/v1`에서 동작합니다.
   - 필요하다면 `Settings > Developer > Server`에서 포트나 인증 토큰을 조정하세요.
3. (선택) 다른 호스트/포트나 토큰을 사용한다면 환경변수를 설정합니다.
   ```bash
   export LM_STUDIO_API_BASE="http://localhost:1234/v1"
   export LM_STUDIO_API_KEY="lm-studio"
   ```
   커맨드라인 옵션 `--api-base`, `--api-key`로도 값을 전달할 수 있습니다.
4. 커맨드라인 게임을 시작합니다.
   ```bash
   python main.py --model "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" --temperature 0.8
   ```
   원하는 경우 모델명이나 temperature 값은 옵션으로 조정할 수 있습니다.

게임 도중 `quit` 혹은 `exit`을 입력하면 세션을 종료합니다.

## 웹 서버 실행
표준 라이브러리 기반 웹 서버를 통해 브라우저에서 동일한 세션을 즐길 수 있습니다.

```bash
python -m server
```

브라우저에서 `http://127.0.0.1:8000`을 열면 대화형 UI가 나타납니다. 서버는 내부적으로 `main.build_game_master`를 사용하므로 커맨드라인과 동일한 환경변수(`TRPG_MODEL`, `TRPG_TEMPERATURE`, `TRPG_API_BASE`, `TRPG_API_KEY`) 설정을 그대로 활용할 수 있습니다.

## MLX Stable Diffusion으로 장면 시각화
Apple Silicon 환경에서 [MLX Stable Diffusion 예제](https://github.com/ml-explore/mlx-examples) 를 설치해 두었다면, 세션마다 현재 장면을 이미지로 렌더링해 웹 UI에서 확인할 수 있습니다.

1. MLX 예제 실행기를 설치합니다. `mlx-examples`는 PyPI에 배포되지 않으므로 저장소를 직접 클론해야 합니다.
   ```bash
   pip install mlx
   git clone https://github.com/ml-explore/mlx-examples.git
   # 실행 스크립트의 경로를 환경변수에 설정합니다.
   export TRPG_MLX_SD_COMMAND="python /path/to/mlx-examples/stable_diffusion/txt2image.py"
   # (mlx-examples를 editable 모드로 설치했다면 기본값 `python -m mlx_examples.stable_diffusion.txt2image`도 사용할 수 있습니다.)
   ```
2. 환경변수로 기능을 켭니다.
   ```bash
   export TRPG_ENABLE_MLX_SD=1
   ```
   필요하다면 실행 커맨드나 모델을 조정할 수 있습니다.
   ```bash
   export TRPG_MLX_SD_COMMAND="python -m mlx_examples.stable_diffusion.txt2image"
   # txt2image는 프롬프트를 위치 인자로 받으며 guidance 파라미터는 --cfg 옵션으로 전달됩니다.
   # (모델 선택이 가능한 버전이라면 'sd' 혹은 'sdxl' 모델 식별자만 지원합니다.)
   export TRPG_MLX_SD_MODEL="sdxl"
   export TRPG_MLX_SD_NEGATIVE_PROMPT="blurry, low quality"
   # 기본적으로 txt2image의 배치 출력을 끄고 (--num_images 1) 변주 수만큼
   # 반복 실행하여 정확히 원하는 장수만 만듭니다.
   # 한 번에 여러 변주(기본 4장)를 만들고 싶다면 값을 조정하세요.
   export TRPG_MLX_SD_VARIATIONS=4
   ```
3. 커맨드라인에서는 `--enable-mlx-sd` 옵션을 사용해도 동일하게 동작합니다.

렌더링에 실패하면 텍스트 로그를 그대로 유지하며, 웹 UI에 오류 메시지가 표시됩니다. 기본 ASCII 요약은 계속 제공됩니다.
