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
