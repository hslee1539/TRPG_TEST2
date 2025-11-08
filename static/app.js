const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sceneImage = document.getElementById("scene-image");
const restartButton = document.getElementById("restart");

let sessionId = null;
let sending = false;

const PLACEHOLDER_IMAGE = sceneImage?.dataset.placeholder
  ? JSON.parse(sceneImage.dataset.placeholder)
  : "";

function appendMessage(role, message) {
  const wrapper = document.createElement("div");
  wrapper.className = `chat-message ${role}`;

  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role === "gm" ? "GM" : "플레이어";
  wrapper.appendChild(roleLabel);

  const text = document.createElement("p");
  text.textContent = message;
  wrapper.appendChild(text);

  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderHistory(history = []) {
  chatLog.innerHTML = "";
  history.forEach((item) => appendMessage(item.role, item.message));
}

function updateScene(sceneText, sceneSvg) {
  if (!sceneImage) {
    return;
  }

  const nextSvg = sceneSvg || PLACEHOLDER_IMAGE;
  sceneImage.innerHTML = nextSvg;
  sceneImage.setAttribute(
    "aria-label",
    sceneText || "현재 장면을 표현한 일러스트"
  );
}

async function startSession() {
  chatInput.value = "";
  chatInput.focus();
  chatLog.innerHTML = "";
  if (sceneImage) {
    sceneImage.innerHTML = PLACEHOLDER_IMAGE;
    sceneImage.setAttribute("aria-label", "현재 장면을 표현한 일러스트");
  }

  try {
    const response = await fetch("/api/session", { method: "POST" });
    if (!response.ok) {
      throw new Error("세션 생성에 실패했습니다.");
    }
    const data = await response.json();
    sessionId = data.sessionId;
    renderHistory(data.history);
    updateScene(data.sceneImageAlt, data.sceneImage);
  } catch (error) {
    appendMessage("gm", error.message || "세션 준비 중 오류가 발생했습니다.");
  }
}

async function sendMessage(message) {
  if (!sessionId || sending) {
    return;
  }

  sending = true;
  appendMessage("player", message);
  chatInput.value = "";
  chatInput.focus();

  try {
    const response = await fetch(`/api/session/${sessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "GM 응답을 불러올 수 없습니다.");
    }

    const data = await response.json();
    renderHistory(data.history);
    updateScene(data.sceneImageAlt, data.sceneImage);
  } catch (error) {
    appendMessage("gm", error.message || "무언가 잘못되었어요. 다시 시도해 주세요.");
  } finally {
    sending = false;
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) {
    return;
  }
  sendMessage(message);
});

restartButton.addEventListener("click", () => {
  startSession();
});

window.addEventListener("DOMContentLoaded", () => {
  startSession();
});
