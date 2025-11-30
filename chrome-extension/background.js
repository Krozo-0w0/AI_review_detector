chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-ai",
    title: "Check if AI-generated",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "check-ai") {
    const selectedText = info.selectionText;

    // console.log("Selected text:", selectedText);
    
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: selectedText })
    })
    .then(res => res.json())
    .then(data => {
      const result = data.prediction;

      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (result) => alert("Model prediction: " + result),
        args: [result]
      });
    });
  }
});
