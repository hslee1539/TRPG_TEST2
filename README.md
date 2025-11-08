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
   - Apple Silicon 환경에서 [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)을 사용하고 있다면, 사전에 `pip install mlx-lm`으로 패키지를 설치한 뒤 원하는 저장소 이름을 `--model` 옵션으로 넘겨 실제 모델을 사용할 수 있습니다.
     ```bash
     python server.py --model "mlx-community/gpt-oss-20b" --temperature 0.6 --max-tokens 400
     ```
     모델 로딩에 성공하면 브라우저 상호작용에서 해당 모델이 바로 응답을 생성합니다. 필수 패키지가 없거나 모델 로딩에 실패하면 친절한 오류 메시지가 출력됩니다.
3. 웹 브라우저에서 [http://localhost:3000](http://localhost:3000)으로 접속하면 새로운 세션이 자동으로 시작되고, 채팅 UI를 통해 TRPG를 진행할 수 있습니다.
   - "새로 시작" 버튼을 누르면 현재 세션이 초기화됩니다.
   - 세션 동안의 대화와 장면 요약은 페이지 내에서 계속 업데이트됩니다.
