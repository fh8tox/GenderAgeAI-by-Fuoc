// Chuyển tab
document.querySelectorAll(".tab-button").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
    });
});

// CAMERA
const webcam = document.getElementById("webcam");
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { webcam.srcObject = stream; })
        .catch(err => console.error("Không thể mở camera:", err));
}

document.getElementById("capture-btn").addEventListener("click", () => {
    const canvas = document.getElementById("snapshot");
    const ctx = canvas.getContext("2d");
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("file", blob, "camera.jpg");
        fetch("/detect_image", { method: "POST", body: formData })
            .then(res => res.blob())
            .then(imageBlob => {
                const url = URL.createObjectURL(imageBlob);
                document.getElementById("result-camera").innerHTML = `<img src="${url}" alt="result">`;
            });
    }, "image/jpeg");
});

// ẢNH
document.getElementById("image-form").addEventListener("submit", e => {
    e.preventDefault();
    const formData = new FormData(e.target);
    fetch("/detect_image", { method: "POST", body: formData })
        .then(res => res.blob())
        .then(imageBlob => {
            const url = URL.createObjectURL(imageBlob);
            document.getElementById("result-image").innerHTML = `<img src="${url}" alt="result">`;
        });
});

// VIDEO
document.getElementById("video-form").addEventListener("submit", e => {
    e.preventDefault();
    const formData = new FormData(e.target);
    fetch("/detect_video", { method: "POST", body: formData })
        .then(res => res.blob())
        .then(videoBlob => {
            const url = URL.createObjectURL(videoBlob);
            document.getElementById("result-video").innerHTML = `<video src="${url}" controls></video>`;
        });
});
