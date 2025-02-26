function waitForCV(callback) {
    if (typeof cv !== "undefined" && cv.getBuildInformation) {
        console.log("✅ OpenCV.js is ready!");
        callback();
    } else {
        console.log("⏳ Waiting for OpenCV to load...");
        setTimeout(() => waitForCV(callback), 100);
    }
}

waitForCV(() => {
    console.log("🚀 Running WebAR SLAM!");

    let video = document.getElementById("video");

    async function startCamera() {
        try {
            let stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = stream;
            console.log("🎥 Camera started!");

            video.addEventListener("loadeddata", () => {
                console.log("🎥 Video is ready!");
                initWebAR(); // ✅ Start WebAR after video loads
            });
        } catch (err) {
            console.error("🚨 Camera access denied:", err);
        }
    }

    startCamera();
});

// ✅ WebAR Initialization
function initWebAR() {
    console.log("🚀 Initializing Three.js WebAR");

    let scene = new THREE.Scene();
    let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    let renderer = new THREE.WebGLRenderer({ alpha: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // ✅ Make sure Three.js is on top of the video
    let threejsContainer = document.createElement("div");
    threejsContainer.id = "threejs-container";
    document.body.appendChild(threejsContainer);
    threejsContainer.appendChild(renderer.domElement);

    // ✅ Add a 3D cube
    let geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    let material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    let cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // ✅ Position cube initially
    cube.position.set(0, 0, -1);
    camera.position.z = 1;

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    trackFeatures(scene, cube, camera);
}



// ✅ SLAM + WebAR Tracking
// Variables for smoothing
let smoothedX = 0, smoothedY = 0;
const SMOOTHING_FACTOR = 0.1;  // Adjust to reduce flickering (lower = smoother)

// ✅ Fix Cube Flickering with Smoothed Position Updates
function trackFeatures(scene, cube, camera) {
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
    canvasOverlay.style.zIndex = "3";
    document.body.appendChild(canvasOverlay);
    let ctxOverlay = canvasOverlay.getContext("2d");

    let prevDescriptors = new cv.Mat();
    let prevKeypoints = new cv.KeyPointVector();

    let smoothedX = 0, smoothedY = 0;
    const SMOOTHING_FACTOR = 0.1;

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

        let videoWidth = video.videoWidth || canvas.width;
        let videoHeight = video.videoHeight || canvas.height;

        ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

        for (let i = 0; i < keypoints.size(); i++) {
            let kp = keypoints.get(i);
            let x = (kp.pt.x / videoWidth) * canvas.width;
            let y = (kp.pt.y / videoHeight) * canvas.height;

            ctxOverlay.beginPath();
            ctxOverlay.arc(x, y, 3, 0, 2 * Math.PI);
            ctxOverlay.fillStyle = "red";
            ctxOverlay.fill();
        }

        if (!prevDescriptors.empty()) {
            let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
            let matches = new cv.DMatchVector();
            bf.match(prevDescriptors, descriptors, matches);

            console.log("🔗 Matched features:", matches.size());

            if (matches.size() > 5) {
                let match = matches.get(0);
                let currPoint = keypoints.get(match.trainIdx).pt;

                let currX = (currPoint.x / videoWidth) * 2 - 1;
                let currY = -(currPoint.y / videoHeight) * 2 + 1;

                // ✅ Apply Smoothing
                smoothedX = smoothedX * (1 - SMOOTHING_FACTOR) + currX * SMOOTHING_FACTOR;
                smoothedY = smoothedY * (1 - SMOOTHING_FACTOR) + currY * SMOOTHING_FACTOR;

                // ✅ Ensure Cube Stays Inside Video Frame
                let maxOffset = 0.8; // Keep cube within visible space
                cube.position.x = Math.max(-maxOffset, Math.min(maxOffset, smoothedX));
                cube.position.y = Math.max(-maxOffset, Math.min(maxOffset, smoothedY));
                cube.position.z = -1.5; // Keep cube closer to the camera

                console.log("🟢 Cube Position Updated:", cube.position.x, cube.position.y);
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



