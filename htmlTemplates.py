css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 10px;
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
    justify-content: flex-end;
}
.chat-message.bot {
    background-color: #475063;
    justify-content: flex-start;
}
.chat-message .message {
    max-width: 80%;
    padding: 10px 15px;
    border-radius: 8px;
    color: #ffffff;
    font-size: 14px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message"><strong>Bot:</strong> {{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message"><strong>You:</strong> {{MSG}}</div>
</div>
'''
