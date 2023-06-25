document.getElementById('queryForm').addEventListener('submit', function(event) {
    event.preventDefault();
  
    const githubLink = document.getElementById('githubLink').value;
    const question = document.getElementById('question').value;
  
    fetch('http://your-api-url/ask', {
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
        document.getElementById('answer').textContent = data.answer;
      });
  });
  