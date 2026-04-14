if (!document.getElementById("yt-floating-btn")) {
  // --- Create Floating Button ---
  const btn = document.createElement("button");
  btn.id = "yt-floating-btn";
  btn.innerText = "TMT";
  Object.assign(btn.style, {
    width: "60px",
    height: "60px",
    border: "none",
    borderRadius: "12px",
    cursor: "pointer",
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: "9999",
    padding: "0",
    color: "white",
    fontWeight: "bold",
    fontSize: "16px",
    textAlign: "center",
    lineHeight: "60px",
    boxShadow: "0px 4px 6px rgba(0,0,0,0.2)",
    background: "linear-gradient(135deg, #8B0000 50%, #000080 50%)"
  });
  document.body.appendChild(btn);

  // --- Create Main Frame ---
  const frame = document.createElement("div");
  frame.id = "yt-floating-frame";
  Object.assign(frame.style, {
    display: "none",
    position: "fixed",
    bottom: "90px",
    right: "20px",
    width: "320px",
    maxHeight: "80vh",
    overflowY: "auto",
    backgroundColor: "#fff",
    border: "1px solid #ccc",
    borderRadius: "10px",
    boxShadow: "0px 4px 12px rgba(0,0,0,0.3)",
    flexDirection: "column",
    zIndex: "9999",
    padding: "10px",
    fontFamily: "Arial, sans-serif",
    display: "flex"
  });

  // --- Header ---
  const header = document.createElement("div");
  Object.assign(header.style, {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px"
  });

  const frameLabel = document.createElement("span");
  frameLabel.innerText = "The Internet Tranquilizer";
  Object.assign(frameLabel.style, { fontWeight: "bold", fontSize: "14px" });

  const closeBtn = document.createElement("button");
  closeBtn.innerText = "✖";
  Object.assign(closeBtn.style, {
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontSize: "16px"
  });

  header.appendChild(frameLabel);
  header.appendChild(closeBtn);
  frame.appendChild(header);

  // --- Sub-Frames ---
  function createSubFrame(title, text = "Loading...") {
    const div = document.createElement("div");
    div.className = "yt-sub-frame clickable";
    Object.assign(div.style, {
      border: "1px solid #ccc",
      borderRadius: "5px",
      padding: "8px",
      marginBottom: "8px",
      backgroundColor: "#f9f9f9"
    });
    div.innerHTML = `<h4 style="margin:0 0 4px 0;">${title}</h4><p style="margin:0;">${text}</p>`;
    return div;
  }

  const frameFact = createSubFrame("Fact Check");
  const frameNeutral = createSubFrame("Neutral Overview");
  const frameMore = createSubFrame("More Info");

  frame.appendChild(frameFact);
  frame.appendChild(frameNeutral);
  frame.appendChild(frameMore);

  // --- Input Box ---
  const inputBox = document.createElement("input");
  inputBox.type = "text";
  inputBox.placeholder = "Want to learn more? Ask TMT here...";
  inputBox.disabled = true;
  Object.assign(inputBox.style, {
    marginTop: "10px",
    width: "100%",
    padding: "8px",
    border: "1px solid #ccc",
    borderRadius: "5px"
  });

  frame.appendChild(inputBox);
  document.body.appendChild(frame);

  // --- Button Click Logic ---
  btn.addEventListener("click", () => {
    const url = window.location.href;
    if (url.includes("youtube.com/watch?v=") || url.includes("youtube.com/shorts/")) {
      frame.style.display = "flex";
      btn.style.display = "none";

      fetch("http://localhost:5000/run-python?url=" + encodeURIComponent(url))
        .then(res => res.text())
        .then(output => {
          const factMatch = output.match(/##FACT CHECK##([\s\S]*?)##END HERE##/);
          const neutralMatch = output.match(/##NEUTRAL OVERVIEW##([\s\S]*?)##END HERE##/);
          const moreMatch = output.match(/##MORE INFO##([\s\S]*?)##END HERE##/);

          frameFact.innerHTML = `<h4>Fact Check</h4><p>${factMatch ? factMatch[1].trim() : "No Fact Check available."}</p>`;
          frameNeutral.innerHTML = `<h4>Neutral Overview</h4><p>${neutralMatch ? neutralMatch[1].trim() : "No Neutral Overview available."}</p>`;
          frameMore.innerHTML = `<h4>More Info</h4><p>${moreMatch ? moreMatch[1].trim() : "No More Info available."}</p>`;

          inputBox.disabled = false;
          inputBox.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
              const userQuery = inputBox.value.trim();
              if (!userQuery) return;
              const contextUrl = `https://mediatranquilzer-d8jvjfofdebtqn7d7wrxij.streamlit.app/?fact=${encodeURIComponent(frameFact.innerText)}&neutral=${encodeURIComponent(frameNeutral.innerText)}&more=${encodeURIComponent(frameMore.innerText)}&query=${encodeURIComponent(userQuery)}`;
              window.open(contextUrl, "_blank");
            }
          });
        })
        .catch(err => console.error("Error:", err));
    } else {
      alert("This product cannot be used on this page.");
    }
  });

  // --- Close Button ---
  closeBtn.addEventListener("click", () => {
    frame.style.display = "none";
    btn.style.display = "block";
  });
}
