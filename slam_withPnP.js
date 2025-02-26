function waitForCV(callback) {
    if (typeof cv !== "undefined" && cv.getBuildInformation) {
        console.log(" OpenCV.js is ready!");
        callback();
    } else {
        console.log(" Waiting for OpenCV to load...");
        setTimeout(() => waitForCV(callback), 100);
    }
}

waitForCV(() => {
    console.log(" Running WebAR SLAM!");

    let video = document.getElementById("video");

    async function startCamera() {
        try {
            let stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = stream;
            console.log(" Camera started!");

            video.addEventListener("loadeddata", () => {
                console.log(" Video is ready!");
                initWebAR();
            });
        } catch (err) {
            console.error(" Camera access denied:", err);
        }
    }

    startCamera();
});

function initWebAR() {
    console.log(" Initializing Three.js WebAR");

    let scene = new THREE.Scene();
    let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    let renderer = new THREE.WebGLRenderer({ alpha: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    
    let threejsContainer = document.createElement("div");
    threejsContainer.id = "threejs-container";
    document.body.appendChild(threejsContainer);
    threejsContainer.appendChild(renderer.domElement);

    let geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    let material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    let cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    cube.position.set(0, 0, -1);
    camera.position.z = 1;

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    trackFeatures(scene, cube, camera);
}

function trackFeatures(scene, cube, camera) {
    let canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    document.body.appendChild(canvas);

    let canvasOverlay = document.getElementById("canvasOverlay");
    if (!canvasOverlay) {
        canvasOverlay = document.createElement("canvas");
        canvasOverlay.id = "canvasOverlay";
        canvasOverlay.width = canvas.width;
        canvasOverlay.height = canvas.height;
        canvasOverlay.style.position = "absolute";
        canvasOverlay.style.top = "0";
        canvasOverlay.style.left = "0";
        canvasOverlay.style.zIndex = "3";
        document.body.appendChild(canvasOverlay);
    }
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
            console.log(" Empty frame detected, skipping...");
            return;
        }

        let keypoints = new cv.KeyPointVector();
        let descriptors = new cv.Mat();
        let detector = new cv.ORB(1000); // Increase features for better tracking
        detector.detectAndCompute(gray, new cv.Mat(), keypoints, descriptors);

        console.log(" Keypoints detected:", keypoints.size());

        let videoWidth = video.videoWidth || canvas.width;
        let videoHeight = video.videoHeight || canvas.height;

        ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

        let objectPoints = [];
        let imagePoints = [];

        for (let i = 0; i < keypoints.size(); i++) {
            let kp = keypoints.get(i);
            let x = (kp.pt.x / videoWidth) * canvas.width;
            let y = (kp.pt.y / videoHeight) * canvas.height;

            ctxOverlay.beginPath();
            ctxOverlay.arc(x, y, 3, 0, 2 * Math.PI);
            ctxOverlay.fillStyle = "red";
            ctxOverlay.fill();

            objectPoints.push([kp.pt.x, kp.pt.y, 0]); 
            imagePoints.push([kp.pt.x, kp.pt.y]);
        }

        if (!prevDescriptors.empty() && objectPoints.length >= 4) {
            let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
            let matches = new cv.DMatchVector();
            bf.match(prevDescriptors, descriptors, matches);

            console.log(" Matched features:", matches.size());

            if (matches.size() >= 4) {
                objectPoints = [];
                imagePoints = [];
            
                for (let i = 0; i < Math.min(30, matches.size()); i++) {
                    let match = matches.get(i);
                    let prevPoint = prevKeypoints.get(match.queryIdx).pt;
                    let currPoint = keypoints.get(match.trainIdx).pt;
            
                    objectPoints.push([prevPoint.x, prevPoint.y, 0]); // Simulated 3D points
                    imagePoints.push([currPoint.x, currPoint.y]);
                }
            
                if (imagePoints.length >= 4) {
                    //  Define `cameraMatrix` Before `solvePnP()`
                    let focalLength = videoWidth; // Approximate focal length
                    let cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [
                        focalLength, 0, videoWidth / 2,
                        0, focalLength, videoHeight / 2,
                        0, 0, 1
                    ]);
            
                    let distCoeffs = new cv.Mat.zeros(4, 1, cv.CV_64F); // Assume no lens distortion
            
                    let objectMat = cv.matFromArray(objectPoints.length, 3, cv.CV_64F, [].concat(...objectPoints));
                    let imageMat = cv.matFromArray(imagePoints.length, 2, cv.CV_64F, [].concat(...imagePoints));
            
                    console.log(" Object Matrix:", objectMat.data64F);
                    console.log(" Image Matrix:", imageMat.data64F);
            
                    let rvec = new cv.Mat(); // Rotation vector
                    let tvec = new cv.Mat(); // Translation vector
            
                    let success = cv.solvePnP(objectMat, imageMat, cameraMatrix, distCoeffs, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);
            
                    if (success) {
                        console.log(" Camera Pose Estimated:", tvec.data64F);
                    
                        let scaleFactor = 0.002; //  Reduce scale to bring cube closer to camera
                        let SMOOTHING_FACTOR = 0.3; //  Increase for smoother motion
                    
                        let targetX = -tvec.data64F[0] * scaleFactor;
                        let targetY = tvec.data64F[1] * scaleFactor;
                        let targetZ = -tvec.data64F[2] * scaleFactor;
                    
                        //  Adjust depth to prevent cube from going outside the view
                        targetZ = Math.max(-3, targetZ); // Keep cube in front of camera
                    
                        //  Apply smoothing for natural movement
                        cube.position.x = cube.position.x * (1 - SMOOTHING_FACTOR) + targetX * SMOOTHING_FACTOR;
                        cube.position.y = cube.position.y * (1 - SMOOTHING_FACTOR) + targetY * SMOOTHING_FACTOR;
                        cube.position.z = cube.position.z * (1 - SMOOTHING_FACTOR) + targetZ * SMOOTHING_FACTOR;
                    
                        console.log(" Cube Adjusted Position:", cube.position.x, cube.position.y, cube.position.z);
                    }
                     else {
                        console.error(" `cv.solvePnP()` FAILED!");
                    }
            
                    rvec.delete();
                    tvec.delete();
                    cameraMatrix.delete();
                    distCoeffs.delete();
                    objectMat.delete();
                    imageMat.delete();
                }
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
