document.getElementById('queryForm').addEventListener('submit', function (event) {
  event.preventDefault();

  const githubLink = document.getElementById('githubLink').value;
  const question = document.getElementById('question').value;

  const submitButton = document.getElementById('submit-button');
  submitButton.setAttribute('disabled', 'disabled');


  fetch('http://localhost:5002/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      githubLink: githubLink,
      question: question,
    }),
  })
    .then(response => response.json())
    .then(data => {
      console.log("Answer: ", data.answer);
      console.log("Chat history: ", data.chat_history);
      let chatHistoryElement = document.getElementById('chat-history');
      data.chat_history.forEach(([question, answer]) => {
        let questionElement = document.createElement('div');
        questionElement.textContent = `Q: ${question}`;
        questionElement.className = 'question';
        chatHistoryElement.appendChild(questionElement);

        let answerElement = document.createElement('div');
        answerElement.innerHTML = `A: ${answer.replace(/\n/g, '<br/>')}`;
        answerElement.className = 'answer';
        chatHistoryElement.appendChild(answerElement);
      });

      submitButton.removeAttribute('disabled');
    })
    .catch(error => {
      console.error('Error:', error);
      submitButton.removeAttribute('disabled');
    });
});
