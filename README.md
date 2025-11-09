# LangChain TRPG Game Master

이 프로젝트는 LangChain을 사용하여 LLM이 게임 마스터 역할을 수행하는 TRPG(테이블탑 롤플레잉 게임) 세션을 실행하는 간단한 커맨드라인 도구입니다.

## 주요 구성 요소
- `trpg.game_master.GameMaster`: LangChain `LLMChain`을 감싸 플레이어 입력과 스토리 진행을 관리합니다.
- `trpg.game_master.GameState`: 현재까지의 사실(facts)을 저장하고 LLM 프롬프트에 제공합니다.
- `trpg.game_master.create_default_game_master`: 기본 시스템 프롬프트와 메모리를 사용하여 따뜻하고 서사적인 진행을 담당하는 게임 마스터를 생성합니다.
- `main.py`: 커맨드라인 인터페이스로, 플레이어 입력을 받아 LLM 응답을 출력합니다.

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
4. 게임을 시작합니다.
   ```bash
   python main.py --model "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" --temperature 0.8
   ```
   원하는 경우 모델명이나 temperature 값은 옵션으로 조정할 수 있습니다.

게임 도중 `quit` 혹은 `exit`을 입력하면 세션을 종료합니다.

## 브라우저에서 플레이하기

CLI 대신 브라우저에서 간단한 인터페이스로 플레이하고 싶다면 Flask 기반의 서버를 실행하세요.

1. 앞선 단계에서 `pip install -r requirements.txt`를 이미 수행했다면 추가 설치는 필요 없습니다.
2. 다음 명령으로 서버를 실행합니다.
   ```bash
   python server.py
   ```
   - LM Studio 서버에서 실행 중인 모델을 그대로 활용하고 싶다면 CLI와 동일하게 모델 이름 및 연결 정보를 전달할 수 있습니다.
     ```bash
     python server.py \
       --model "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" \
       --temperature 0.6 \
       --max-tokens 400 \
       --api-base "http://localhost:1234/v1" \
       --api-key "lm-studio"
     ```
     API 서버 주소와 키는 생략 시 환경변수(`LM_STUDIO_API_BASE`, `LM_STUDIO_API_KEY`)나 기본값을 따릅니다. 서버는 단일 LLM 인스턴스를 재사용하므로 브라우저 새 세션에서도 모델이 즉시 응답합니다.
3. 웹 브라우저에서 [http://localhost:3000](http://localhost:3000)으로 접속하면 새로운 세션이 자동으로 시작되고, 채팅 UI를 통해 TRPG를 진행할 수 있습니다.
   - "새로 시작" 버튼을 누르면 현재 세션이 초기화됩니다.
   - 세션 동안의 대화와 장면 요약은 페이지 내에서 계속 업데이트됩니다.
   - 게임 마스터의 묘사, 플레이어의 행동, 최근 사실을 조합한 프롬프트로 MLX Stable Diffusion이 양자화된 모델을 통해 PNG 이미지를 생성하고, 해당 이미지를 브라우저에 즉시 렌더링합니다.
   - 장면 생성이 실패하면 API는 500 상태 코드와 함께 오류 메시지를 반환하며, 이때는 Stable Diffusion 구성을 확인해야 합니다.

### MLX Stable Diffusion으로 장면 그리기

`server.py`는 기본적으로 키워드 기반 SVG 데이터 URI를 그리는 로컬 생성기를 사용합니다. 실제 Stable Diffusion 출력을 사용하려면 Apple MLX 환경이 필요하며 아래 절차를 따르세요.

1. macOS에서 기본 의존성을 설치합니다.
   ```bash
   pip install mlx Pillow numpy
   ```
2. MLX 예제 저장소를 준비합니다. `mlx-examples`는 패키징되어 있지 않으므로 클론 후 경로를 직접 추가해야 합니다.
   ```bash
   git clone https://github.com/ml-explore/mlx-examples.git
   cd mlx-examples
   pip install -r stable_diffusion/requirements.txt
   export PYTHONPATH="$(pwd):${PYTHONPATH}"
   ```
   루트 디렉터리에 `requirements.txt` 파일이 없으므로 Stable Diffusion 예제 폴더의 요구 사항 파일을 직접 설치해야 합니다. 위 명령은 예제 저장소 의존성을 설치하고 현재 셸 세션에서 `mlx_examples` 모듈을 찾을 수 있도록 경로를 노출합니다.
3. MLX 예제 저장소의 안정화된 양자화 모델을 다운로드하거나 직접 변환합니다. 예를 들어 [`mlx-examples`](https://github.com/ml-explore/mlx-examples)의 `stable_diffusion` 스크립트로 `--quantize` 옵션을 사용해 모델을 준비할 수 있습니다.
4. 서버 실행 시 Stable Diffusion 관련 옵션을 전달합니다. `--sd-model`을 생략하면
   LM Studio 모델을 지정했을 때 기본값으로 `mlx-community/stable-diffusion-v1-5-diffusers`
   리포지토리를 사용하려고 시도합니다. 미리 다운로드한 경로를 `MLX_SD_MODEL`
   환경변수에 지정해도 됩니다.
   ```bash
   python server.py \
     --model "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" \
     --sd-model "/path/to/stable-diffusion-quantized" \
     --sd-steps 35 \
     --sd-guidance 7.0 \
     --sd-negative "blurry, text, watermark" \
     --sd-width 768 \
     --sd-height 512
   ```
   - `--sd-model`은 양자화된 Stable Diffusion 모델 폴더 혹은 체크포인트 경로입니다.
   - `--sd-steps`, `--sd-guidance`, `--sd-negative`, `--sd-width`, `--sd-height`, `--sd-seed`로 세부 파라미터를 조정할 수 있습니다.
   - `--sd-no-quantize`를 지정하면 양자화 옵션을 비활성화할 수 있지만, 기본값은 양자화를 사용하도록 설정되어 있습니다.
   - Stable Diffusion을 사용하지 않으려면 `--sd-disable` 플래그를 추가해 SVG 키워드 생성기로 되돌릴 수 있습니다.

Stable Diffusion을 활성화하면 서버는 단일 파이프라인 인스턴스를 재사용해 매 요청마다 장면 이미지를 생성합니다. 출력은 PNG 데이터 URI 형태로 브라우저에 전달됩니다.

## 테스트

모든 기능을 검증하려면 다음 명령을 사용하세요.
```bash
pytest
```
