function waitForCV(callback) {
    if (window.cvReady && typeof cv !== "undefined") {
        console.log("OpenCV.js is ready!");
        callback();
    } else {
        console.log("Waiting for OpenCV to load...");
        setTimeout(() => waitForCV(callback), 100);
    }
}

waitForCV(() => {
    console.log("Running SLAM!");

    let video = document.getElementById("video");
    let canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");

    async function initSLT() {
        let stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;

        video.onloadeddata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            trackFeatures();
        };
    }

    function trackFeatures() {
        let src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
        let gray = new cv.Mat();
        let canvasOverlay = document.createElement("canvas");  // Create overlay canvas
        let ctxOverlay = canvasOverlay.getContext("2d");
    
        canvasOverlay.width = canvas.width;
        canvasOverlay.height = canvas.height;
        canvasOverlay.style.position = "absolute";
        canvasOverlay.style.top = "0";
        canvasOverlay.style.left = "0";
        document.body.appendChild(canvasOverlay);
    
        setInterval(() => {
            let keypoints = new cv.KeyPointVector();  // ✅ Ensure keypoints are re-created every frame
            let detector = new cv.ORB();
    
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);  // Clear previous circles
    
            src.data.set(ctx.getImageData(0, 0, canvas.width, canvas.height).data);
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            detector.detect(gray, keypoints);  // ✅ Detect features before deleting keypoints
    
            console.log("Tracking", keypoints.size(), "features.");
    
            // Draw circles on detected keypoints
            for (let i = 0; i < keypoints.size(); i++) {
                let kp = keypoints.get(i);
                ctxOverlay.beginPath();
                ctxOverlay.arc(kp.pt.x, kp.pt.y, 5, 0, 2 * Math.PI);  // Draw circle
                ctxOverlay.strokeStyle = "red";
                ctxOverlay.lineWidth = 2;
                ctxOverlay.stroke();
            }
    
            detector.delete();
            keypoints.delete();
        }, 100);
    }
    initSLT();
});
