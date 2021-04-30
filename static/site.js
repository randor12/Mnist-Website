/**
 * Website JavaScript Functions Go Here
 */
 var canvas;
 var context;
 var clickX = new Array();
 var clickY = new Array();
 var clickDrag = new Array();
 var paint = false;
 var cursorColor = "#FFFFFF";

 /**
  * Make a canvas to draw on
  **/
 function drawCanvas() {

      // Get the canvas element
     canvas = document.getElementById('canvas');
     context = document.getElementById('canvas').getContext("2d");
     // Create a mouse down event - allows for drawing the object
     $('#canvas').mousedown(function (e) {
         var mouseX = e.pageX - this.offsetLeft;
         var mouseY = e.pageY - this.offsetTop;

         paint = true;
         addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
         draw();
     });
     // Allow for the drawing while moving the mouse
     $('#canvas').mousemove(function (e) {
         if (paint) {
             addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
             draw();
         }
     });
     // On mouse up, stop drawing
     $('#canvas').mouseup(function (e) {
         paint = false;
     });
 }

 /**
  * Saves the click postition
  * @args x: This is the x position
  * @args y: this is the y position
  * @args dragging: this is a boolean operation on if the user is moving the mouse or not
  **/
 function addClick(x, y, dragging) {
     clickX.push(x);
     clickY.push(y);
     clickDrag.push(dragging);
 }

 /**
  * Clear the canvas and draw the picture
  **/
 function draw() {

     context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
     // set the paint color, size and shape
     context.strokeStyle = cursorColor;
     context.lineJoin = "round";
     context.lineWidth = 27;
     for (var i = 0; i < clickX.length; i++) {
       // Draw the "paint" on the canvas
       context.beginPath();
       if (clickDrag[i] && i) {
         context.moveTo(clickX[i - 1], clickY[i - 1]);
       } else {
         context.moveTo(clickX[i] - 1, clickY[i]);
       }
       context.lineTo(clickX[i], clickY[i]);
       context.closePath();
       context.stroke();
     }
 }

 function clear_canvas() {
   console.log('Clearing');
   clickX = new Array();
   clickY = new Array();
   clickDrag = new Array();

   canvas = document.getElementById('canvas');
   context = document.getElementById('canvas').getContext('2d');
   console.log(context);

   context.clearRect(0, 0, context.canvas.width, context.canvas.height); // clear the canvas
   console.log(context);
 }

 /**
  * Encodes the image into a base 64 string.
  * This will then add the string to an hidden tag of the form so Flask can reach it.
  **/
 function send_pic() {
     var image = new Image();
     var url = document.getElementById('url');
     image.id = "pic";
     image.src = canvas.toDataURL();
     url.value = image.src;
 }
