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

    addMessage(message, true);
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;
    addTyping();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: fullMessage })
        });
        const data = await response.json();
        removeTyping();
        addMessage(data.response, false);
    } catch (error) {
        removeTyping();
        addMessage('Error connecting to server. Please try again.', false);
    }

    btn.disabled = false;
    input.focus();
}
