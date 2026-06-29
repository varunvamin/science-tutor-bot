let chatHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    // Load theme preference
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
        const themeBtn = document.getElementById('themeBtn');
        if(themeBtn) themeBtn.innerHTML = '☀️';
    }
});

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function clearChat() {
    // A full reload ensures the state is completely reset and the welcome message returns
    window.location.reload();
}

function toggleTheme() {
    const body = document.body;
    const btn = document.getElementById('themeBtn');
    if (body.classList.contains('dark-theme')) {
        body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light');
        if (btn) btn.innerHTML = '🌙';
    } else {
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
        if (btn) btn.innerHTML = '☀️';
    }
}

function exportChat() {
    if (chatHistory.length === 0) return alert("No chat history to export yet!");
    let content = "🔬 Science Tutor Bot - Chat Export\n\n";
    chatHistory.forEach(msg => {
        const role = msg.role === 'user' ? 'You' : 'Tutor';
        // Clean up format tags from user messages
        const text = msg.content.replace(/ \(.*? format\)$/, '');
        content += `${role}:\n${text}\n\n`;
    });
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'science-tutor-notes.txt';
    a.click();
    URL.revokeObjectURL(url);
}

function readAloud(text, btn) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        // Remove HTML tags for reading
        const plainText = text.replace(/<[^>]+>/g, '');
        const utterance = new SpeechSynthesisUtterance(plainText);
        
        const originalText = btn.innerHTML;
        btn.innerHTML = '🔊';
        btn.style.opacity = '1';
        
        utterance.onend = () => {
            btn.innerHTML = originalText;
            btn.style.opacity = '';
        };
        
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Text-to-speech is not supported in your browser.");
    }
}

function sendExample(text) {
    document.getElementById('messageInput').value = text;
    sendMessage();
}

function addMessage(text, isUser) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();

    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '👤' : '🔬';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (isUser) {
        bubble.textContent = text;
    } else {
        // Format bot response
        if (text === 'This chatbot only answers science-related questions.') {
            bubble.innerHTML = `<span class="rejected">⚠️ ${text}</span>`;
        } else {
            // Convert bullet points to HTML
            const formatted = text
                .replace(/•\s(.+)/g, '<li>$1</li>')
                .replace(/(\d+)\.\s(.+)/g, '<li>$2</li>');
            bubble.innerHTML = formatted.includes('<li>') 
                ? `<ul>${formatted}</ul>` 
                : `<p>${text}</p>`;

            // Add copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'action-btn copy-btn';
            copyBtn.innerHTML = '📋';
            copyBtn.title = 'Copy to clipboard';
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(text);
                copyBtn.innerHTML = '✅';
                setTimeout(() => copyBtn.innerHTML = '📋', 2000);
            };
            bubble.appendChild(copyBtn);

            // Add text-to-speech button
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'action-btn tts-btn';
            ttsBtn.innerHTML = '🔈';
            ttsBtn.title = 'Read aloud';
            ttsBtn.onclick = () => readAloud(bubble.innerHTML, ttsBtn);
            bubble.appendChild(ttsBtn);
        }
    }

    div.appendChild(avatar);
    div.appendChild(bubble);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function addTyping() {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = 'message bot';
    div.id = 'typing';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🔬';

    const bubble = document.createElement('div');
    bubble.className = 'bubble typing';
    bubble.innerHTML = '<span></span><span></span><span></span>';

    div.appendChild(avatar);
    div.appendChild(bubble);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const typing = document.getElementById('typing');
    if (typing) typing.remove();
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const btn = document.getElementById('sendBtn');
    const format = document.getElementById('formatSelect').value;
    const message = input.value.trim();

    if (!message) return;

    // Add format hint to message
    const fullMessage = message + (format !== 'bullet' ? ` (${format} format)` : '');

    // Add to history
    chatHistory.push({ role: 'user', content: fullMessage });

    addMessage(message, true);
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;
    addTyping();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: fullMessage, history: chatHistory })
        });
        const data = await response.json();
        
        // Add bot response to history
        chatHistory.push({ role: 'assistant', content: data.response });
        
        removeTyping();
        addMessage(data.response, false);
    } catch (error) {
        removeTyping();
        addMessage('Error connecting to server. Please try again.', false);
    }

    btn.disabled = false;
    input.focus();
}
