window.addEventListener("load", () => {
  const canvas = document.querySelector("#canvas");
  const ctx = canvas.getContext("2d");

  // Resizing
  canvas.height = 500; //window.innerHeight;
  canvas.width = 900;//window.innerWidth;

  //variables
  let painting = false;

  function startPosition(e){
    painting = true;
    draw(e) // to enable user to draw points
  }
  function finishPosition(){
    painting = false;
    ctx.beginPath();
  }

  function draw(e){
    if(!painting) return;
    ctx.lineWidth = 30;
    ctx.lineCap = "round";
//  ctx.strokeStyle = "red";

    ctx.lineTo(e.clientX, e.clientY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX, e.clientY);
  }
  //Eventlisteners
  canvas.addEventListener("mousedown", startPosition);
  canvas.addEventListener("mouseup", finishPosition);
  canvas.addEventListener("mousemove", draw);
});
