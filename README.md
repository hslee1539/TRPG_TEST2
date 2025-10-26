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
   pip install langchain openai
   ```
2. OpenAI API 키를 환경변수로 설정합니다.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. 게임을 시작합니다.
   ```bash
   python main.py --model gpt-3.5-turbo --temperature 0.8
   ```
   원하는 경우 모델명이나 temperature 값은 옵션으로 조정할 수 있습니다.

게임 도중 `quit` 혹은 `exit`을 입력하면 세션을 종료합니다.
