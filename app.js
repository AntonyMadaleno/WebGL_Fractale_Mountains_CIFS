var vertexShaderText = `
precision highp float;

attribute vec3 vertPosition;
attribute vec3 vertNormal;
attribute vec3 vertColor;

uniform mat4 mWorld;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat3 u_normalMatrix;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_color;
varying vec3 v_eyeDir;

void main() {
  // Transform the vertex position and normal
  vec4 position = mProj * mView * mWorld * vec4(vertPosition, 1.0);
  gl_Position = position;
  
  // Calculate the vertex's eye-space position
  v_position = position.xyz;
  
  // Transform the normal to eye space
  v_normal = normalize(u_normalMatrix * vertNormal);
  
  // Calculate the eye direction (towards the camera)
  v_eyeDir = normalize(-v_position);
  v_color = vertColor;
}
`

var fragmentShaderText = `
precision highp float;

uniform vec3 u_ambientColor;
uniform vec3 u_diffuseColor;
uniform vec3 u_specularColor;
uniform float u_shininess;
uniform vec3 u_lightPosition;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_eyeDir;
varying vec3 v_color;

void main() {
  // Calculate the normalized light direction
  vec3 lightDir = normalize(u_lightPosition - v_position);
  
  // Calculate the reflection direction
  vec3 reflectDir = reflect(-lightDir, v_normal);
  
  // Calculate the diffuse component
  float diffuseFactor = max(dot(v_normal, lightDir), 0.0);
  vec3 diffuse = v_color * diffuseFactor;
  
  // Calculate the specular component
  float specularFactor = pow(max(dot(reflectDir, v_eyeDir), 0.0), u_shininess);
  vec3 specular = v_color * specularFactor;
  
  // Combine ambient, diffuse, and specular lighting
  vec3 ambient = u_ambientColor;
  vec3 result = v_color * 0.1 + diffuse + specular;
  
  gl_FragColor = vec4(result, 1.0);
}
`
function generateMountain(k = 8, delta = 0.3, range = 2, base_size = 0) {

  let n = 2**base_size;
  let m = 2**base_size;
  let heightMap = Matrix.random(n+1,m+1, 0,1);

  for (let c = 0; c < k; c++) {
    delta = delta / range;
    n = n * 2;
    m = m * 2;
    let tmp = new Matrix(n + 1, m + 1);

    for (let i = 0; i < heightMap.rows; i++) {
      for (let j = 0; j < heightMap.cols; j++) {
        tmp.setAt(i * 2, j * 2, heightMap.at(i, j));
      }
    }

    for (let i = 0; i < heightMap.cols - 1; i++) {
      for (let j = 0; j < heightMap.rows - 1; j++) {
        const a = heightMap.at(i, j);
        const b = heightMap.at(i, j + 1);
        const c = heightMap.at(i + 1, j + 1);
        const d = heightMap.at(i + 1, j);

        const value = (a + b + c + d) / 4 + delta * (Math.random() * 2 - 1);
        tmp.setAt(1 + i * 2, 1 + j * 2, value);
      }
    }

    // FIRST ROW
    for (let i = 0; i < heightMap.cols - 1; i++) {
      const a = heightMap.at(i, 0);
      const b = heightMap.at(i + 1, 0);
      const c = tmp.at(1 + i * 2, 1);

      const value = (a + b + c) / 3 + delta * ( Math.random() * 2 - 1);
      tmp.setAt(1 + i * 2, 0, value);
    }

    // LAST ROW
    for (let i = 0; i < heightMap.cols - 1; i++) {
      const a = heightMap.at(i, m / 2);
      const b = heightMap.at(i + 1, m / 2);
      const c = tmp.at(1 + i * 2, m - 1);

      const value = (a + b + c) / 3 + delta * (Math.random() * 2 - 1);
      tmp.setAt(1 + i * 2, m, value);
    }

    // FIRST COLUMN
    for (let i = 0; i < heightMap.rows - 1; i++) {
      const a = heightMap.at(0, i);
      const b = heightMap.at(0, i + 1);
      const c = tmp.at(1, 1 + i * 2);

      const value = (a + b + c) / 3 + delta * (Math.random() * 2 - 1);
      tmp.setAt(0, 1 + i * 2, value);
    }

    // LAST COLUMN
    for (let i = 0; i < heightMap.rows - 1; i++) {
      const a = heightMap.at(n / 2, i);
      const b = heightMap.at(n / 2, i + 1);
      const c = tmp.at(n - 1, 1 + i * 2);

      const value = (a + b + c) / 3 + delta * (Math.random() * 2 - 1);
      tmp.setAt(n, 1 + i * 2, value);
    }

    for (let j = 1; j < tmp.rows - 2; j++) {
      for (let i = 1; i < tmp.cols - 2; i++) {
        if (i % 2 !== j % 2) {
          const a = tmp.at(i - 1, j);
          const b = tmp.at(i, j - 1);
          const c = tmp.at(i + 1, j);
          const d = tmp.at(i, j + 1);

          const value = (a + b + c + d) / 4 + delta * (Math.random() * 2 - 1);

          tmp.setAt(i, j, value);
        }
      }
    }

    heightMap = tmp.copy();
  }

  return heightMap;
}


let white = [1.0, 1.0, 1.0];
let grey = [0.6, 0.6, 0.6];
let forest = [0.1, 0.75, 0.2];

function calculateNormal(heightMap, i, j) {
  // Define the neighbor offsets
  const offsets = [
    { x: -1, y: -1 }, // Top-left
    { x: 0, y: -1 },  // Top
    { x: 1, y: -1 },  // Top-right
    { x: -1, y: 0 },  // Left
    { x: 1, y: 0 },   // Right
    { x: -1, y: 1 },  // Bottom-left
    { x: 0, y: 1 },   // Bottom
    { x: 1, y: 1 }    // Bottom-right
  ];

  // Initialize variables for the sums of heights in the x and z directions
  let sumX = 0;
  let sumZ = 0;

  // Iterate through the neighboring points
  for (const offset of offsets) {
    const ni = i + offset.x;
    const nj = j + offset.y;

    // Check if the indices are within bounds
    if (ni >= 0 && ni < heightMap.cols && nj >= 0 && nj < heightMap.rows) {
      // Get the height at the neighboring point
      const height = heightMap.at(ni, nj);

      // Update the sums of heights in the x and z directions
      sumX += offset.x * height;
      sumZ += offset.y * height;
    }
  }

  // Calculate the normal vector components
  const normalX = -sumX; // Negate for correct direction
  const normalY = 1.0;   // Vertical component
  const normalZ = -sumZ; // Negate for correct direction

  // Normalize the normal vector
  const length = Math.sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
  return [normalX / length, normalY / length, normalZ / length];
}


function generateVertices(heightMap)
{
	vertices = [];
	indices = [];
	dx = 10.0 / heightMap.cols;
	dz = 10.0 / heightMap.rows;

	for (let j = 0; j < heightMap.rows; j++)
	{
		for (let i = 0; i < heightMap.cols; i++)
		{
			let x = i*dx - 5.0; let z = j*dz - 5.0;
			let y = heightMap.at(i,j);

			vertices.push(x);
			vertices.push(Math.max(y, 0.25) * 2 );
			vertices.push(z);

			// Calculate normals based on neighboring vertices
		  const normal = calculateNormal(heightMap, i, j);
		  vertices.push(normal[0]);
		  vertices.push(normal[1]);
		  vertices.push(normal[2]);

			if (y <= 0.1)
			{
				vertices.push(0.0);
				vertices.push(0.1);
				vertices.push(0.5);
			}
			else if (y <= 0.25)
			{
				vertices.push(0.1);
				vertices.push(0.3);
				vertices.push(0.8);
			}
			else if (y <= 0.75)
			{
				vertices.push(forest[0]);
				vertices.push(forest[1]);
				vertices.push(forest[2]);
			}
			else if (y <= 0.9)
			{
				vertices.push(grey[0]);
				vertices.push(grey[1]);
				vertices.push(grey[2]);
			}
			else if (y > 0.9)
			{
				vertices.push(white[0]);
				vertices.push(white[1]);
				vertices.push(white[2]);
			}

			if (i < heightMap.cols -1 && j < heightMap.rows -1)
			{
				indices.push( (j) * heightMap.cols 	 + i+1 );
				indices.push( (j+1) * heightMap.cols + i+1 );
				indices.push( (j) * heightMap.cols 	 + i );

				indices.push( (j+1) * heightMap.cols + i+1 );
				indices.push( (j+1) * heightMap.cols + i );
				indices.push( (j) * heightMap.cols 	 + i );
			}

		}
	}

	return [vertices, indices];
}

fps_display = document.getElementById("fps");

var canvas = document.getElementById('game-surface');
const contextAttributes = { antialiasing: false, depth: true, depthPrecision: 'highp' };
const gl = canvas.getContext('webgl', contextAttributes);
const ext = gl.getExtension("OES_element_index_uint");

var InitDemo = function () {

	let heightMap = generateMountain(k = 6, delta = 0.25, range = 2, base_size = 3);
	let res = generateVertices(heightMap);

	if (!gl) {
		console.log('WebGL not supported, falling back on experimental-webgl');
		gl = canvas.getContext('experimental-webgl');
	}

	if (!gl) {
		alert('Your browser does not support WebGL');
	}

	gl.clearColor(0.00, 0.00, 0.00, 1.0);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	gl.enable(gl.DEPTH_TEST);
	gl.enable(gl.CULL_FACE);
	gl.frontFace(gl.CCW);
	gl.cullFace(gl.FRONT);

	//
	// Create shaders
	// 
	var vertexShader = gl.createShader(gl.VERTEX_SHADER);
	var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);

	gl.shaderSource(vertexShader, vertexShaderText);
	gl.shaderSource(fragmentShader, fragmentShaderText);

	gl.compileShader(vertexShader);
	if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
		console.error('ERROR compiling vertex shader!', gl.getShaderInfoLog(vertexShader));
		return;
	}

	gl.compileShader(fragmentShader);
	if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
		console.error('ERROR compiling fragment shader!', gl.getShaderInfoLog(fragmentShader));
		return;
	}

	var program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		console.error('ERROR linking program!', gl.getProgramInfoLog(program));
		return;
	}
	gl.validateProgram(program);
	if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
		console.error('ERROR validating program!', gl.getProgramInfoLog(program));
		return;
	}

	//
	// Create buffer
	//
	var boxVertices = res[0];
	var boxIndices = res[1];

	var boxVertexBufferObject = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, boxVertexBufferObject);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(boxVertices), gl.STATIC_DRAW);

	var boxIndexBufferObject = gl.createBuffer();
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boxIndexBufferObject);
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(boxIndices), gl.STATIC_DRAW);

	var positionAttribLocation = gl.getAttribLocation(program, 'vertPosition');
	var normalAttribLocation = gl.getAttribLocation(program, 'vertNormal');
	var colorAttribLocation = gl.getAttribLocation(program, 'vertColor');
	gl.vertexAttribPointer(
		positionAttribLocation, // Attribute location
		3, // Number of elements per attribute
		gl.FLOAT, // Type of elements
		gl.FALSE,
		9 * Float32Array.BYTES_PER_ELEMENT, // Size of an individual vertex
		0 // Offset from the beginning of a single vertex to this attribute
	);
	gl.vertexAttribPointer(
		normalAttribLocation, // Attribute location
		3, // Number of elements per attribute
		gl.FLOAT, // Type of elements
		gl.FALSE,
		9 * Float32Array.BYTES_PER_ELEMENT, // Size of an individual vertex
		3 * Float32Array.BYTES_PER_ELEMENT// Offset from the beginning of a single vertex to this attribute
	);
	gl.vertexAttribPointer(
		colorAttribLocation, // Attribute location
		3, // Number of elements per attribute
		gl.FLOAT, // Type of elements
		gl.FALSE,
		9 * Float32Array.BYTES_PER_ELEMENT, // Size of an individual vertex
		6 * Float32Array.BYTES_PER_ELEMENT // Offset from the beginning of a single vertex to this attribute
	);

	gl.enableVertexAttribArray(positionAttribLocation);
	gl.enableVertexAttribArray(normalAttribLocation);
	gl.enableVertexAttribArray(colorAttribLocation);

	// Tell OpenGL state machine which program should be active.
	gl.useProgram(program);

	const matWorldUniformLocation = gl.getUniformLocation(program, 'mWorld');
	const matViewUniformLocation = gl.getUniformLocation(program, 'mView');
	const matProjUniformLocation = gl.getUniformLocation(program, 'mProj');

	var worldMatrix = new Float32Array(16);
	var viewMatrix = new Float32Array(16);
	var projMatrix = new Float32Array(16);
	mat4.identity(worldMatrix);
	mat4.lookAt(viewMatrix, [0, 5, -4], [0, 1, 0], [0, 1, 0]);
	mat4.perspective(projMatrix, glMatrix.toRadian(45), canvas.clientWidth / canvas.clientHeight, 0.1, 1000.0);

	gl.uniformMatrix4fv(matWorldUniformLocation, gl.FALSE, worldMatrix);
	gl.uniformMatrix4fv(matViewUniformLocation, gl.FALSE, viewMatrix);
	gl.uniformMatrix4fv(matProjUniformLocation, gl.FALSE, projMatrix);

	const uAmbientColor = gl.getUniformLocation(program, 'u_ambientColor');
	const uDiffuseColor = gl.getUniformLocation(program, 'u_diffuseColor');
	const uSpecularColor = gl.getUniformLocation(program, 'u_specularColor');
	const uShininess = gl.getUniformLocation(program, 'u_shininess');
	const uLightPosition = gl.getUniformLocation(program, 'u_lightPosition');
	const uNormalMatrix = gl.getUniformLocation(program, 'u_normalMatrix');

	// Set the uniform values
	gl.uniform3fv(uAmbientColor, [0.2, 0.2, 0.2]); // Ambient color as an example
	gl.uniform3fv(uDiffuseColor, [1.0, 1.0, 1.0]); // Diffuse color as an example
	gl.uniform3fv(uSpecularColor, [1.0, 1.0, 1.0]); // Specular color as an example
	gl.uniform1f(uShininess, 30.0); // Shininess value as an example
	gl.uniform3fv(uLightPosition, [-5, 20.0, -5]); // Light position as an example
	// Calculate the normal matrix from the model-view matrix
	const normalMatrix = mat3.create();
	mat3.normalFromMat4(normalMatrix, viewMatrix);

	// Set the normal matrix uniform
	gl.uniformMatrix3fv(uNormalMatrix, false, normalMatrix);

	var xRotationMatrix = new Float32Array(16);
	var yRotationMatrix = new Float32Array(16);

	//
	// Main render loop
	//
	var identityMatrix = new Float32Array(16);
	mat4.identity(identityMatrix);
	var angle = 0;
	var t0 = performance.now();
	var t1 = performance.now();
	var loop = function () {
		t1 = performance.now();
		fps_display.innerText = (1000 / (t1 - t0)).toFixed(0) + " fps";
		t0 = performance.now();
		angle = performance.now() / 4000 / 6 * 2 * Math.PI;
		mat4.rotate(yRotationMatrix, identityMatrix, angle, [0, 1, 0]);
		mat4.rotate(xRotationMatrix, identityMatrix, 0, [1, 0, 0]);
		mat4.mul(worldMatrix, yRotationMatrix, xRotationMatrix);
		gl.uniformMatrix4fv(matWorldUniformLocation, gl.FALSE, worldMatrix);

		gl.clearColor(0.00, 0.00, 0.00, 1.0);
		gl.clear(gl.DEPTH_BUFFER_BIT | gl.COLOR_BUFFER_BIT);
		gl.drawElements(gl.TRIANGLES, boxIndices.length, gl.UNSIGNED_INT, 0);

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);
};