function waitForCV(callback) {
    if (typeof cv !== "undefined" && cv.getBuildInformation) {
        console.log("✅ OpenCV.js is ready!");
        callback();
    } else {
        console.log("⏳ Waiting for OpenCV to load...");
        setTimeout(() => waitForCV(callback), 100);
    }
}

// ✅ Ensure `trackFeatures()` is defined before `startCamera()`
function trackFeatures() {
    let video = document.getElementById("video");

    let canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    document.body.appendChild(canvas);

    let canvasOverlay = document.createElement("canvas");
    canvasOverlay.id = "canvasOverlay";
    canvasOverlay.width = canvas.width;
    canvasOverlay.height = canvas.height;
    canvasOverlay.style.position = "absolute";
    canvasOverlay.style.top = "0";
    canvasOverlay.style.left = "0";
    document.body.appendChild(canvasOverlay);
    let ctxOverlay = canvasOverlay.getContext("2d");

    let prevDescriptors = new cv.Mat();
    let prevKeypoints = new cv.KeyPointVector();

    function processFrame() {
        if (video.readyState < 2) {
            console.log("⏳ Waiting for video frame...");
            return;
        }

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        let src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
        src.data.set(imageData.data);
        let gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        if (gray.rows === 0 || gray.cols === 0) {
            console.log("🚨 Empty frame detected, skipping...");
            return;
        }

        let keypoints = new cv.KeyPointVector();
        let descriptors = new cv.Mat();
        let detector = new cv.ORB();
        detector.detectAndCompute(gray, new cv.Mat(), keypoints, descriptors);

        console.log("🔍 Keypoints detected:", keypoints.size());

        ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

        for (let i = 0; i < keypoints.size(); i++) {
            let kp = keypoints.get(i);
            let x = kp.pt.x * (canvas.width / video.videoWidth);
            let y = kp.pt.y * (canvas.height / video.videoHeight);

            ctxOverlay.beginPath();
            ctxOverlay.arc(x, y, 5, 0, 2 * Math.PI);
            ctxOverlay.strokeStyle = "red";
            ctxOverlay.lineWidth = 2;
            ctxOverlay.stroke();
        }

        if (!prevDescriptors.empty()) {
            let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
            let matches = new cv.DMatchVector();
            bf.match(prevDescriptors, descriptors, matches);

            console.log("🔗 Matched features:", matches.size());

            for (let i = 0; i < matches.size(); i++) {
                let match = matches.get(i);
                let prevPoint = prevKeypoints.get(match.queryIdx).pt;
                let currPoint = keypoints.get(match.trainIdx).pt;

                let prevX = prevPoint.x * (canvas.width / video.videoWidth);
                let prevY = prevPoint.y * (canvas.height / video.videoHeight);
                let currX = currPoint.x * (canvas.width / video.videoWidth);
                let currY = currPoint.y * (canvas.height / video.videoHeight);

                ctxOverlay.beginPath();
                ctxOverlay.moveTo(prevX, prevY);
                ctxOverlay.lineTo(currX, currY);
                ctxOverlay.strokeStyle = "blue";
                ctxOverlay.lineWidth = 1;
                ctxOverlay.stroke();
            }

            matches.delete();
            bf.delete();
        }

        prevDescriptors.delete();
        prevDescriptors = descriptors.clone();
        prevKeypoints.delete();
        prevKeypoints = keypoints.clone();

        keypoints.delete();
        descriptors.delete();
        detector.delete();
    }

    setInterval(processFrame, 100);
}

waitForCV(() => {
    console.log("🚀 Running Feature Matching SLT!");

    let video = document.getElementById("video");

    async function startCamera() {
        try {
            let stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = stream;
            console.log("🎥 Camera started!");

            video.addEventListener("loadeddata", () => {
                console.log("🎥 Video is ready!");
                trackFeatures(); // ✅ Now `trackFeatures()` is defined before being called
            });
        } catch (err) {
            console.error("🚨 Camera access denied:", err);
        }
    }

    startCamera();
});
