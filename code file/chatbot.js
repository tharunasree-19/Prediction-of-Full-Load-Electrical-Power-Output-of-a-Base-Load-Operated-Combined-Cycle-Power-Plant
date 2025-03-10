// JavaScript code to send a message and handle the response
function sendMessage() {
    var message = document.getElementById("userMessage").value;

    if (message.trim() === "") {
        alert("Please enter a message!");
        return;
    }

    fetch('/chat_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Check the response here for debugging
        console.log(data);
        document.getElementById("chatResponse").innerText = data.response;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("chatResponse").innerText = "⚠ An error occurred while processing your request.";
    });
}
