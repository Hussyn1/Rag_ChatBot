// ============================================
// NeuralQuery — Chat Interface Logic
// ============================================

const API_BASE = "";

// State
let chatHistory = [];
let isStreaming = false;

// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const chatContainer = document.getElementById("chatContainer");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const welcomeScreen = document.getElementById("welcomeScreen");
const suggestionChips = document.getElementById("suggestionChips");
const statsText = document.getElementById("statsText");

// ============================================
// Initialization
// ============================================

document.addEventListener("DOMContentLoaded", () => {
    loadStats();
    setupEventListeners();
    messageInput.focus();
});

function setupEventListeners() {
    // Send button
    sendBtn.addEventListener("click", handleSend);

    // Textarea input
    messageInput.addEventListener("input", () => {
        autoResizeTextarea();
        sendBtn.disabled = !messageInput.value.trim() || isStreaming;
    });

    // Enter to send, Shift+Enter for newline
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (messageInput.value.trim() && !isStreaming) {
                handleSend();
            }
        }
    });

    // Clear chat
    clearChatBtn.addEventListener("click", clearChat);

    // Suggestion chips
    suggestionChips.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip");
        if (chip) {
            const query = chip.dataset.query;
            messageInput.value = query;
            autoResizeTextarea();
            handleSend();
        }
    });
}

// ============================================
// Stats
// ============================================

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        statsText.textContent = `${data.total_chunks} chunks indexed`;
    } catch (err) {
        statsText.textContent = "Connecting...";
    }
}

// ============================================
// Chat Logic
// ============================================

async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isStreaming) return;

    // Hide welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }

    // Add user message
    appendMessage("user", message);
    chatHistory.push({ role: "user", content: message });

    // Clear input
    messageInput.value = "";
    autoResizeTextarea();
    sendBtn.disabled = true;
    isStreaming = true;

    // Show typing indicator
    const typingEl = showTypingIndicator();

    try {
        // Call the streaming API
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-10), // Last 10 messages
            }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Remove typing indicator
        typingEl.remove();

        // Create assistant message container
        const { messageEl, bubbleEl } = createAssistantMessage();
        let fullContent = "";
        let sources = [];

        // Read the SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    try {
                        const chunk = JSON.parse(jsonStr);

                        if (chunk.type === "content") {
                            fullContent += chunk.content;
                            bubbleEl.innerHTML = renderMarkdown(fullContent);
                            scrollToBottom();
                        } else if (chunk.type === "sources") {
                            sources = chunk.sources || [];
                        } else if (chunk.type === "error") {
                            bubbleEl.classList.add("error-bubble");
                            bubbleEl.textContent = chunk.content;
                        } else if (chunk.type === "done") {
                            // Append sources if available
                            if (sources.length > 0) {
                                appendSources(messageEl.querySelector(".message-content"), sources);
                            }
                        }
                    } catch (parseErr) {
                        // Skip malformed chunks
                    }
                }
            }
        }

        // Save assistant message to history
        chatHistory.push({ role: "assistant", content: fullContent });

    } catch (err) {
        typingEl.remove();
        appendErrorMessage(err.message);
    } finally {
        isStreaming = false;
        sendBtn.disabled = !messageInput.value.trim();
        messageInput.focus();
    }
}

// ============================================
// Message Rendering
// ============================================

function appendMessage(role, content) {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;

    const avatarEl = document.createElement("div");
    avatarEl.className = "message-avatar";
    avatarEl.textContent = role === "user" ? "Y" : "AI";

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";

    const bubbleEl = document.createElement("div");
    bubbleEl.className = "message-bubble";

    if (role === "user") {
        bubbleEl.textContent = content;
    } else {
        bubbleEl.innerHTML = renderMarkdown(content);
    }

    contentEl.appendChild(bubbleEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(contentEl);
    chatMessages.appendChild(messageEl);
    scrollToBottom();
}

function createAssistantMessage() {
    const messageEl = document.createElement("div");
    messageEl.className = "message assistant";

    const avatarEl = document.createElement("div");
    avatarEl.className = "message-avatar";
    avatarEl.textContent = "AI";

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";

    const bubbleEl = document.createElement("div");
    bubbleEl.className = "message-bubble";

    contentEl.appendChild(bubbleEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(contentEl);
    chatMessages.appendChild(messageEl);
    scrollToBottom();

    return { messageEl, bubbleEl };
}

function appendSources(contentEl, sources) {
    const container = document.createElement("div");
    container.className = "sources-container";

    const label = document.createElement("div");
    label.className = "sources-label";
    label.textContent = "Sources";
    container.appendChild(label);

    // De-duplicate sources by name
    const seen = new Set();
    for (const src of sources) {
        if (seen.has(src.source)) continue;
        seen.add(src.source);

        const badge = document.createElement("span");
        badge.className = "source-badge";
        badge.innerHTML = `
            📄 ${src.source.replace('.md', '').replace(/_/g, ' ')}
            <span class="source-relevance">${Math.round(src.relevance * 100)}%</span>
        `;
        container.appendChild(badge);
    }

    contentEl.appendChild(container);
    scrollToBottom();
}

function appendErrorMessage(errorText) {
    const messageEl = document.createElement("div");
    messageEl.className = "message assistant";

    const avatarEl = document.createElement("div");
    avatarEl.className = "message-avatar";
    avatarEl.textContent = "AI";

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";

    const bubbleEl = document.createElement("div");
    bubbleEl.className = "message-bubble error-bubble";
    bubbleEl.textContent = `⚠️ ${errorText}. Please check your API key and try again.`;

    contentEl.appendChild(bubbleEl);
    messageEl.appendChild(avatarEl);
    messageEl.appendChild(contentEl);
    chatMessages.appendChild(messageEl);
    scrollToBottom();
}

// ============================================
// Typing Indicator
// ============================================

function showTypingIndicator() {
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.innerHTML = `
        <div class="message-avatar" style="background: var(--gradient-subtle); border: 1px solid var(--glass-border); color: var(--accent-indigo);">AI</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
    return el;
}

// ============================================
// Markdown Rendering
// ============================================

function renderMarkdown(text) {
    if (!text) return "";

    let html = text;

    // Escape HTML (except our markdown)
    html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Code blocks (```...```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold (**...**)
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic (*...*)
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

    // Headings
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

    // Unordered lists
    html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    // Line breaks (double newline = paragraph)
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    // Wrap in paragraph if not already wrapped
    if (!html.startsWith("<")) {
        html = `<p>${html}</p>`;
    }

    return html;
}

// ============================================
// Utility Functions
// ============================================

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + "px";
}

function clearChat() {
    chatHistory = [];
    chatMessages.innerHTML = "";

    // Restore welcome screen
    const welcome = document.createElement("div");
    welcome.className = "welcome-screen";
    welcome.id = "welcomeScreen";
    welcome.innerHTML = `
        <div class="welcome-icon">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="30" stroke="url(#welcomeGrad2)" stroke-width="2" opacity="0.3"/>
                <circle cx="32" cy="32" r="20" stroke="url(#welcomeGrad2)" stroke-width="2" opacity="0.5"/>
                <circle cx="32" cy="32" r="10" fill="url(#welcomeGrad2)" opacity="0.8"/>
                <defs>
                    <linearGradient id="welcomeGrad2" x1="2" y1="2" x2="62" y2="62">
                        <stop stop-color="#818cf8"/>
                        <stop offset="1" stop-color="#c084fc"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <h2>Welcome to NeuralQuery</h2>
        <p>Your AI-powered knowledge assistant for Machine Learning, Deep Learning, RAG, and Context Engineering. Ask me anything!</p>
        <div class="suggestion-chips" id="suggestionChips">
            <button class="chip" data-query="What is the difference between supervised and unsupervised learning?">
                ML: Supervised vs Unsupervised
            </button>
            <button class="chip" data-query="Explain the transformer architecture and self-attention mechanism">
                DL: Transformer Architecture
            </button>
            <button class="chip" data-query="How does the RAG pipeline work? Explain the indexing and retrieval phases.">
                RAG: Pipeline Architecture
            </button>
            <button class="chip" data-query="What is context engineering and how does it differ from prompt engineering?">
                Context vs Prompt Engineering
            </button>
        </div>
    `;
    chatMessages.appendChild(welcome);

    // Re-bind suggestion chips
    const newChips = welcome.querySelector(".suggestion-chips");
    newChips.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip");
        if (chip) {
            messageInput.value = chip.dataset.query;
            autoResizeTextarea();
            handleSend();
        }
    });

    messageInput.focus();
}
