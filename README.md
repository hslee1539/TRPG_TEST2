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
   - 게임 마스터의 묘사나 플레이어의 행동에 따라 장면 일러스트가 마을, 숲, 던전, 성 등으로 자동 전환됩니다.
