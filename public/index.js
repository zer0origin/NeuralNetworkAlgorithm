var canvas,
    ctx,
    flag = false,
    prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    dot_flag = false;

var x = "black",
    y = 1; // Lowered to 1 so it matches the 28x28 resolution perfectly
function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;
    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

function color(obj) {
    switch (obj.id) {
        case "green":
            x = "green";
            break;
        case "blue":
            x = "blue";
            break;
        case "red":
            x = "red";
            break;
        case "yellow":
            x = "yellow";
            break;
        case "orange":
            x = "orange";
            break;
        case "black":
            x = "black";
            break;
        case "white":
            x = "white";
            break;
    }
    if (x == "white") y = 3; // Eraser size scaled down for 28x28 grid
    else y = 1;
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = x;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.clearRect(0, 0, w, h);

    for (let x = 0; x < 10; x++) {
        console.log(`answer-${x}`)
        document.getElementById(`answer-${x}`).style.backgroundColor = `rgb(255, 255, 255)`
    }
}

// Helper function to scale mouse coordinates down to the 28x28 space
function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((e.clientX - rect.left) / rect.width) * canvas.width,
        y: ((e.clientY - rect.top) / rect.height) * canvas.height
    };
}

function findxy(res, e) {
    if (res == 'down') {
        const coords = getCanvasCoordinates(e);
        prevX = currX;
        prevY = currY;
        currX = coords.x;
        currY = coords.y;
        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(Math.floor(currX), Math.floor(currY), y, y);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            const coords = getCanvasCoordinates(e);
            prevX = currX;
            prevY = currY;
            currX = coords.x;
            currY = coords.y;
            draw();
        }
    }
}

function submit() {
    let pixels = canvasToMnistRaw();
    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    const raw = JSON.stringify({
        "bytes": Array.from(pixels)
    });

    const requestOptions = {
        method: "POST",
        headers: myHeaders,
        body: raw,
        redirect: "follow"
    };

    fetch("/guess", requestOptions)
        .then((response) => response.json())
        .then((resp) => {
            for (x in resp.results) {
                console.log(`answer-${x}`)
                document.getElementById(`answer-${x}`).style.backgroundColor = `rgb(0, ${255 * Number(resp.results[x][0])}, 0)`
            }
        })
        .catch((error) => console.error(error));
}

function canvasToMnistRaw() {
    const ctx = canvas.getContext('2d');

    // This returns a Uint8ClampedArray of exactly 3136 bytes (RGBA)
    const imgData = ctx.getImageData(0, 0, 28, 28);
    const rawBytes = imgData.data;

    let pixelIndex = 0;
    const mnistPixels = new Uint8Array(28 * 28);
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let index = (((y * 28) + x) * 4) + 3

            const currentBrightness = rawBytes[index];
            const invertedValue = currentBrightness;
            mnistPixels[pixelIndex] = invertedValue;
            pixelIndex++;
        }
    }

    console.log(mnistPixels);

    return mnistPixels;
}
